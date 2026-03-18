/// Training Loop & Replay Buffer — Tiered Architecture
///
/// Implements the Self-Play Tournament Pipeline:
/// 1. Hall of Fame (HoF): Top-N elite programs with ELO ratings
/// 2. Replay Buffer (Training Pool): Diverse winning sequences for training
/// 3. Tournament Pipeline: Champions vs Gladiators with ELO ranking
/// 4. Curriculum Learning: Arena scales up over generations
/// 5. Chaos Mode: Temperature spike on stagnation
///
/// Pipeline per generation:
///   Champion Selection → Gen Gladiators → Battle Royale → ELO Update
///   → HoF Promotion → Buffer Push → Weighted Training → Chaos Check

use std::path::Path;

use burn::prelude::*;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::arena::{Arena, ArenaConfig};
use crate::constraint::{
    self, decode_tokens, tokens_to_ids, Token, END_TOKEN,
};
use crate::elo::{self, DEFAULT_ELO};
use crate::transformer::{SubleqTransformer, SubleqTransformerConfig, training_step};

// ============================================================================
// Hall of Fame — The Elite Registry
// ============================================================================

/// A single entry in the Hall of Fame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoFEntry {
    /// Token ID sequence (for training)
    pub token_ids: Vec<usize>,
    /// Decoded SUBLEQ program (for arena battles)
    pub program: Vec<i64>,
    /// ELO rating
    pub elo: f64,
    /// Generation when this entry was inducted
    pub generation_born: usize,
}

/// The Hall of Fame — stores the top-N elite programs
#[derive(Debug, Serialize, Deserialize)]
pub struct HallOfFame {
    pub entries: Vec<HoFEntry>,
    pub max_size: usize,
}

impl HallOfFame {
    pub fn new(max_size: usize) -> Self {
        HallOfFame {
            entries: Vec::new(),
            max_size,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Try to promote a new entry into the Hall of Fame.
    /// Returns true if the entry was inducted (either room available or beat weakest).
    pub fn try_promote(&mut self, entry: HoFEntry) -> bool {
        if self.entries.len() < self.max_size {
            log::info!(
                "⭐ [HoF] New inductee! ELO={:.0}, gen={} (filling slot {}/{})",
                entry.elo, entry.generation_born, self.entries.len() + 1, self.max_size
            );
            self.entries.push(entry);
            return true;
        }

        // Find the weakest champion
        let (weakest_idx, weakest_elo) = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.elo.partial_cmp(&b.elo).unwrap())
            .map(|(i, e)| (i, e.elo))
            .unwrap();

        if entry.elo > weakest_elo {
            log::info!(
                "👑 [HoF] Dethroned! New ELO={:.0} > Old ELO={:.0} (gen {}→{})",
                entry.elo, weakest_elo,
                self.entries[weakest_idx].generation_born, entry.generation_born
            );
            self.entries[weakest_idx] = entry;
            return true;
        }

        false
    }

    /// Select N random champions for gatekeeper duty.
    /// Returns their indices and cloned entries.
    pub fn select_champions(&self, count: usize) -> Vec<(usize, HoFEntry)> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let mut rng = rand::thread_rng();
        let count = count.min(self.entries.len());
        let indices: Vec<usize> = (0..self.entries.len()).collect();
        let selected: Vec<&usize> = indices.choose_multiple(&mut rng, count).collect();
        selected
            .into_iter()
            .map(|&i| (i, self.entries[i].clone()))
            .collect()
    }

    /// Update ELO of an existing entry by index
    pub fn update_elo(&mut self, index: usize, new_elo: f64) {
        if index < self.entries.len() {
            self.entries[index].elo = new_elo;
        }
    }

    /// Get all token sequences (for weighted batch training)
    pub fn all_token_ids(&self) -> Vec<&Vec<usize>> {
        self.entries.iter().map(|e| &e.token_ids).collect()
    }

    /// Save to JSON
    pub fn save(&self, path: &str) {
        match std::fs::File::create(path) {
            Ok(file) => {
                if let Err(e) = serde_json::to_writer(file, self) {
                    log::error!("Failed to save Hall of Fame: {}", e);
                } else {
                    log::info!("[HoF] Saved ({} entries) to {}", self.entries.len(), path);
                }
            }
            Err(e) => log::error!("Failed to create HoF file: {}", e),
        }
    }

    /// Load from JSON. Returns empty HoF if file doesn't exist.
    pub fn load(path: &str, default_max_size: usize) -> Self {
        if let Ok(file) = std::fs::File::open(path) {
            if let Ok(hof) = serde_json::from_reader::<_, HallOfFame>(file) {
                log::info!("[HoF] Loaded {} entries from {}", hof.entries.len(), path);
                return hof;
            } else {
                log::warn!("Failed to parse HoF file, starting fresh");
            }
        }
        Self::new(default_max_size)
    }
}

// ============================================================================
// Replay Buffer — Training Pool with weighted sampling
// ============================================================================

/// Replay buffer storing winning token sequences.
/// Serializable to JSON for persistence across runs.
#[derive(Debug, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// Stored winning programs as token ID sequences
    pub sequences: Vec<Vec<usize>>,
    /// Maximum buffer capacity
    pub max_capacity: usize,
}

impl ReplayBuffer {
    pub fn new(max_capacity: usize) -> Self {
        ReplayBuffer {
            sequences: Vec::new(),
            max_capacity,
        }
    }

    /// Add a winning sequence to the buffer.
    /// If buffer is full, remove the oldest entry (FIFO).
    pub fn push(&mut self, token_ids: Vec<usize>) {
        if self.sequences.len() >= self.max_capacity {
            self.sequences.remove(0);
        }
        self.sequences.push(token_ids);
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Save buffer to a JSON file
    pub fn save(&self, path: &str) {
        match std::fs::File::create(path) {
            Ok(file) => {
                if let Err(e) = serde_json::to_writer(file, self) {
                    log::error!("Failed to save replay buffer: {}", e);
                } else {
                    log::info!("Replay buffer saved to {}", path);
                }
            }
            Err(e) => log::error!("Failed to create replay buffer file: {}", e),
        }
    }

    /// Load buffer from a JSON file. Returns a new empty buffer if file doesn't exist.
    pub fn load(path: &str, default_capacity: usize) -> Self {
        if let Ok(file) = std::fs::File::open(path) {
            if let Ok(buffer) = serde_json::from_reader::<_, ReplayBuffer>(file) {
                log::info!(
                    "Loaded replay buffer from {} ({} sequences)",
                    path,
                    buffer.sequences.len()
                );
                return buffer;
            } else {
                log::warn!("Failed to parse replay buffer file, starting fresh");
            }
        }
        Self::new(default_capacity)
    }

    /// Create a **weighted** training batch mixing HoF and Training Pool data.
    ///
    /// `hof_sequences`: token sequences from Hall of Fame champions
    /// `hof_ratio`: fraction of the batch to fill from HoF (e.g., 0.7 = 70%)
    ///
    /// Teacher forcing: input = tokens[:-1], target = tokens[1:]
    pub fn make_weighted_batch<B: Backend>(
        &self,
        hof_sequences: &[&Vec<usize>],
        device: &B::Device,
        max_seq_len: usize,
        batch_size: usize,
        hof_ratio: f64,
    ) -> Option<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
        let total_available = self.sequences.len() + hof_sequences.len();
        if total_available == 0 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let actual_batch_size = batch_size.min(total_available);

        // Calculate how many from each source
        let hof_count = if hof_sequences.is_empty() {
            0
        } else {
            ((actual_batch_size as f64 * hof_ratio) as usize).min(hof_sequences.len())
        };
        let pool_count = if self.sequences.is_empty() {
            0
        } else {
            (actual_batch_size - hof_count).min(self.sequences.len())
        };

        // Sample from each source
        let mut selected: Vec<&Vec<usize>> = Vec::with_capacity(hof_count + pool_count);

        if hof_count > 0 {
            let hof_sampled: Vec<&&Vec<usize>> =
                hof_sequences.choose_multiple(&mut rng, hof_count).collect();
            selected.extend(hof_sampled.into_iter().map(|x| *x));
        }
        if pool_count > 0 {
            let pool_sampled: Vec<&Vec<usize>> =
                self.sequences.choose_multiple(&mut rng, pool_count).collect();
            selected.extend(pool_sampled);
        }

        if selected.is_empty() {
            return None;
        }

        let final_batch_size = selected.len();

        // Find max length in this sample
        let actual_max = selected.iter().map(|s| s.len()).max().unwrap_or(2);
        let seq_len = actual_max.min(max_seq_len).max(2);

        // Build padded input and target arrays
        let target_len = seq_len - 1;
        let mut input_data = vec![0i64; final_batch_size * target_len];
        let mut target_data = vec![0i64; final_batch_size * target_len];

        for (i, seq) in selected.iter().enumerate() {
            for j in 0..target_len {
                let idx = i * target_len + j;
                if j < seq.len() {
                    input_data[idx] = seq[j] as i64;
                }
                if j + 1 < seq.len() {
                    target_data[idx] = seq[j + 1] as i64;
                } else {
                    target_data[idx] = END_TOKEN as i64;
                }
            }
        }

        let input_tensor = Tensor::<B, 1, Int>::from_data(&input_data[..], device)
            .reshape([final_batch_size, target_len]);
        let target_tensor = Tensor::<B, 1, Int>::from_data(&target_data[..], device)
            .reshape([final_batch_size, target_len]);

        Some((input_tensor, target_tensor))
    }
}

// ============================================================================
// Curriculum Learning — Arena scales up over generations
// ============================================================================

/// A single stage in the curriculum
#[derive(Debug, Clone)]
pub struct CurriculumStage {
    /// Apply this stage until this generation (exclusive)
    pub until_generation: usize,
    /// Arena memory size for this stage
    pub arena_memory_size: usize,
    /// Arena slot size for this stage
    pub gladiator_slot_size: usize,
    /// Max battle rounds for this stage
    pub max_rounds: u32,
}

/// Get the ArenaConfig for a given generation based on curriculum stages.
fn curriculum_arena_config(generation: usize, stages: &[CurriculumStage]) -> ArenaConfig {
    for stage in stages {
        if generation < stage.until_generation {
            return ArenaConfig {
                memory_size: stage.arena_memory_size,
                gladiator_slot_size: stage.gladiator_slot_size,
                max_rounds: stage.max_rounds,
            };
        }
    }
    // Past all stages: use the last stage's config
    if let Some(last) = stages.last() {
        ArenaConfig {
            memory_size: last.arena_memory_size,
            gladiator_slot_size: last.gladiator_slot_size,
            max_rounds: last.max_rounds,
        }
    } else {
        ArenaConfig::default()
    }
}

// ============================================================================
// Checkpoint management
// ============================================================================

const MODEL_CHECKPOINT_PATH: &str = "checkpoints/model";
const BUFFER_CHECKPOINT_PATH: &str = "checkpoints/replay_buffer.json";
const HOF_CHECKPOINT_PATH: &str = "checkpoints/hall_of_fame.json";

/// Save model checkpoint to disk
fn save_model_checkpoint<B: Backend>(model: &SubleqTransformer<B>) {
    let _ = std::fs::create_dir_all("checkpoints");
    match model
        .clone()
        .save_file(MODEL_CHECKPOINT_PATH, &CompactRecorder::new())
    {
        Ok(_) => log::info!("Model checkpoint saved"),
        Err(e) => log::error!("Failed to save model checkpoint: {}", e),
    }
}

/// Try to load model checkpoint from disk.
fn load_model_checkpoint<B: Backend>(
    config: &SubleqTransformerConfig,
    device: &B::Device,
) -> Option<SubleqTransformer<B>> {
    let checkpoint_file = format!("{}.mpk.gz", MODEL_CHECKPOINT_PATH);
    if !Path::new(&checkpoint_file).exists() {
        return None;
    }
    let model = config.init::<B>(device);
    match model.load_file(MODEL_CHECKPOINT_PATH, &CompactRecorder::new(), device) {
        Ok(loaded) => {
            log::info!("Model checkpoint loaded from {}", checkpoint_file);
            Some(loaded)
        }
        Err(e) => {
            log::warn!("Failed to load model checkpoint: {}", e);
            None
        }
    }
}

// ============================================================================
// Evolution Configuration
// ============================================================================

/// Configuration for the evolution loop
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Number of NEW gladiators to generate per battle
    pub gladiators_per_battle: usize,
    /// Number of generations to run
    pub num_generations: usize,
    /// Number of training steps per generation
    pub train_steps_per_gen: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Temperature for token sampling (base, may be overridden by chaos)
    pub temperature: f64,
    /// Maximum program length (in tokens)
    pub max_program_tokens: usize,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Training batch size
    pub train_batch_size: usize,
    /// How often to save checkpoints (every N generations)
    pub checkpoint_interval: usize,

    // --- Hall of Fame ---
    /// Max entries in the Hall of Fame
    pub hof_max_size: usize,
    /// Number of champions to pull from HoF per battle
    pub champions_per_battle: usize,
    /// Ratio of HoF data in training batches (0.0–1.0)
    pub hof_train_ratio: f64,

    // --- Curriculum Learning ---
    /// Curriculum stages (arena scales up over time)
    pub curriculum_stages: Vec<CurriculumStage>,

    // --- Chaos Mode ---
    /// Trigger chaos after this many stagnant generations
    pub chaos_stagnation_threshold: usize,
    /// Temperature override during chaos mode
    pub chaos_temperature: f64,

    /// Transformer config
    pub transformer_config: SubleqTransformerConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        EvolutionConfig {
            gladiators_per_battle: 8,
            num_generations: 100,
            train_steps_per_gen: 5,
            learning_rate: 1e-3,
            temperature: 1.0,
            max_program_tokens: 64,
            buffer_capacity: 1000,
            train_batch_size: 32,
            checkpoint_interval: 10,

            hof_max_size: 20,
            champions_per_battle: 3,
            hof_train_ratio: 0.7,

            curriculum_stages: vec![
                CurriculumStage {
                    until_generation: 20,
                    arena_memory_size: 256,
                    gladiator_slot_size: 64,
                    max_rounds: 1_000,
                },
                CurriculumStage {
                    until_generation: 50,
                    arena_memory_size: 512,
                    gladiator_slot_size: 64,
                    max_rounds: 10_000,
                },
                CurriculumStage {
                    until_generation: usize::MAX,
                    arena_memory_size: 1024,
                    gladiator_slot_size: 64,
                    max_rounds: 50_000,
                },
            ],

            chaos_stagnation_threshold: 5,
            chaos_temperature: 2.0,

            transformer_config: SubleqTransformerConfig::new(),
        }
    }
}

// ============================================================================
// Evolution Loop — The Tournament Pipeline
// ============================================================================

/// Run the full evolution loop with Tournament Pipeline.
///
/// Pipeline per generation:
///   1. Curriculum: pick arena config
///   2. Champion Selection: pull N champions from HoF
///   3. Generation: Transformer generates new gladiators
///   4. Battle Royale: champions + gladiators fight
///   5. ELO Update: compute new ratings
///   6. Promotion: new gladiators that beat champions → HoF
///   7. Buffer Push: winners into Training Pool
///   8. Weighted Training: 70% HoF + 30% Pool
///   9. Chaos Check: stagnation → temperature spike
///  10. Checkpoint
pub fn evolution_loop<B: AutodiffBackend>(
    config: EvolutionConfig,
    device: &B::Device,
) {
    log::info!("=== THE SUBLEQ SPIRE — TOURNAMENT EDITION ===");

    // --- Load or initialize model ---
    let mut model = if let Some(loaded) =
        load_model_checkpoint::<B>(&config.transformer_config, device)
    {
        log::info!("Resuming from checkpoint!");
        loaded
    } else {
        log::info!("Initializing fresh Transformer model...");
        config.transformer_config.init::<B>(device)
    };

    // --- Initialize optimizer ---
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init::<B, SubleqTransformer<B>>();

    // --- Load or initialize replay buffer ---
    let mut buffer = ReplayBuffer::load(BUFFER_CHECKPOINT_PATH, config.buffer_capacity);

    // --- Load or initialize Hall of Fame ---
    let mut hof = HallOfFame::load(HOF_CHECKPOINT_PATH, config.hof_max_size);

    // Seed buffer if empty (first run)
    if buffer.is_empty() {
        log::info!("Seeding replay buffer with random programs...");
        seed_random_programs(&mut buffer, 50, config.max_program_tokens);
    }

    log::info!(
        "Config: {} new gladiators + {} champions per battle, {} gens",
        config.gladiators_per_battle,
        config.champions_per_battle,
        config.num_generations
    );
    log::info!(
        "HoF: max={}, train_ratio={:.0}%, chaos_threshold={} gens",
        config.hof_max_size,
        config.hof_train_ratio * 100.0,
        config.chaos_stagnation_threshold
    );
    log::info!("Starting evolution...");

    let mut stagnation_counter: usize = 0;

    for gen in 0..config.num_generations {
        log::info!("━━━ Generation {} ━━━", gen + 1);

        // === Phase 1: Curriculum — pick arena config ===
        let arena_config = curriculum_arena_config(gen, &config.curriculum_stages);
        if gen == 0
            || (gen > 0
                && curriculum_arena_config(gen - 1, &config.curriculum_stages).memory_size
                    != arena_config.memory_size)
        {
            log::info!(
                "📚 [Curriculum] Arena: {} cells, {} rounds",
                arena_config.memory_size, arena_config.max_rounds
            );
        }

        // === Phase 2: Champion Selection ===
        let champions = hof.select_champions(config.champions_per_battle);
        let num_champions = champions.len();
        if num_champions > 0 {
            let avg_elo: f64 =
                champions.iter().map(|(_, e)| e.elo).sum::<f64>() / num_champions as f64;
            log::info!(
                "🛡️  [HoF] Summoned {} champions (avg ELO={:.0})",
                num_champions, avg_elo
            );
        }

        // === Phase 3: Chaos check → determine temperature ===
        let effective_temp = if stagnation_counter >= config.chaos_stagnation_threshold {
            log::info!(
                "🌀 [CHAOS] Stagnation detected ({} gens)! Temperature → {:.1}",
                stagnation_counter, config.chaos_temperature
            );
            config.chaos_temperature
        } else {
            config.temperature
        };

        // === Phase 4: Generate new gladiators ===
        // Champion programs go first, then new gladiators
        let mut all_programs: Vec<Vec<i64>> = Vec::new();
        let mut all_token_sequences: Vec<Vec<Token>> = Vec::new();
        let mut all_token_ids: Vec<Vec<usize>> = Vec::new();
        let mut initial_elos: Vec<f64> = Vec::new();
        // Track which HoF index each champion corresponds to (-1 for non-champions)
        let mut champion_hof_indices: Vec<Option<usize>> = Vec::new();

        // Add champions
        for (hof_idx, entry) in &champions {
            all_programs.push(entry.program.clone());
            all_token_ids.push(entry.token_ids.clone());
            all_token_sequences.push(Vec::new()); // placeholder — champions don't need this
            initial_elos.push(entry.elo);
            champion_hof_indices.push(Some(*hof_idx));
        }

        // Generate new gladiators from Transformer
        let mut gen_count = 0;
        for _ in 0..config.gladiators_per_battle {
            let tokens = model.generate(device, effective_temp, config.max_program_tokens);
            let program = decode_tokens(&tokens);
            if program.len() >= 3 {
                all_programs.push(program);
                all_token_ids.push(tokens_to_ids(&tokens));
                all_token_sequences.push(tokens);
                initial_elos.push(DEFAULT_ELO);
                champion_hof_indices.push(None);
                gen_count += 1;
            }
        }

        // Fill with random if needed (chaos mode generates more random bots)
        let random_target = if stagnation_counter >= config.chaos_stagnation_threshold {
            config.gladiators_per_battle // Double the random bots during chaos
        } else {
            config.gladiators_per_battle.saturating_sub(gen_count)
        };
        for _ in 0..random_target.saturating_sub(gen_count) {
            let (prog, tokens) = generate_random_program(config.max_program_tokens);
            all_token_ids.push(tokens_to_ids(&tokens));
            all_token_sequences.push(tokens);
            all_programs.push(prog);
            initial_elos.push(DEFAULT_ELO);
            champion_hof_indices.push(None);
        }

        let total_fighters = all_programs.len();
        log::info!(
            "⚔️  Battle: {} fighters ({} champions + {} new, temp={:.2})",
            total_fighters, num_champions, total_fighters - num_champions, effective_temp
        );

        // === Phase 5: Battle Royale ===
        let mut arena = Arena::new(arena_config);
        arena.spawn(&all_programs);
        let result = arena.run_battle();

        log::info!(
            "Battle result: {} rounds, {} survivors",
            result.total_rounds, result.survivors
        );

        // === Phase 6: ELO Update ===
        let updated_elos = elo::compute_battle_elos(
            total_fighters,
            &result.elimination_order,
            result.winner_index,
            &initial_elos,
        );

        // Log ELO changes
        for i in 0..total_fighters {
            let tag = if champion_hof_indices[i].is_some() {
                "Champion"
            } else {
                "Gladiator"
            };
            let delta = updated_elos[i] - initial_elos[i];
            let arrow = if delta >= 0.0 { "↑" } else { "↓" };
            log::debug!(
                "  {} #{}: ELO {:.0} → {:.0} ({}{:.0})",
                tag, i, initial_elos[i], updated_elos[i], arrow, delta.abs()
            );
        }

        // === Phase 7: Update champion ELOs in HoF & Promote new gladiators ===
        let mut hof_changed = false;

        // Update existing champions' ELO
        for (i, hof_idx_opt) in champion_hof_indices.iter().enumerate() {
            if let Some(hof_idx) = hof_idx_opt {
                hof.update_elo(*hof_idx, updated_elos[i]);
            }
        }

        // Try to promote new gladiators
        for i in num_champions..total_fighters {
            // Only promote non-champions with decent ELO
            if updated_elos[i] > DEFAULT_ELO + 10.0 {
                let entry = HoFEntry {
                    token_ids: all_token_ids[i].clone(),
                    program: all_programs[i].clone(),
                    elo: updated_elos[i],
                    generation_born: gen,
                };
                if hof.try_promote(entry) {
                    hof_changed = true;
                }
            }
        }

        // === Phase 8: Push winners into Replay Buffer ===
        if let Some(winner_idx) = result.winner_index {
            log::info!(
                "🏆 Winner: Fighter {} (ELO={:.0}, {})",
                winner_idx,
                updated_elos[winner_idx],
                if champion_hof_indices[winner_idx].is_some() { "Champion" } else { "New" }
            );
            buffer.push(all_token_ids[winner_idx].clone());
        } else {
            log::info!("💀 No winner — all eliminated.");
            if let Some(&last_eliminated) = result.elimination_order.last() {
                buffer.push(all_token_ids[last_eliminated].clone());
                log::info!(
                    "Using last eliminated (Fighter {}, ELO={:.0}) as surrogate.",
                    last_eliminated, updated_elos[last_eliminated]
                );
            }
        }

        // Also push any new gladiator that gained ELO (diverse training data)
        for i in num_champions..total_fighters {
            if updated_elos[i] > DEFAULT_ELO {
                buffer.push(all_token_ids[i].clone());
            }
        }

        // === Phase 9: Weighted Training ===
        let hof_seqs = hof.all_token_ids();
        if buffer.len() + hof_seqs.len() >= 2 {
            for step in 0..config.train_steps_per_gen {
                if let Some((input, target)) = buffer.make_weighted_batch::<B>(
                    &hof_seqs,
                    device,
                    config.max_program_tokens,
                    config.train_batch_size,
                    config.hof_train_ratio,
                ) {
                    let (loss, grads) = training_step(&model, input, target);
                    let loss_val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(config.learning_rate, model, grads);

                    if step == 0 {
                        log::info!(
                            "📉 Loss: {:.4} (batch={}, HoF ratio={:.0}%)",
                            loss_val, config.train_batch_size, config.hof_train_ratio * 100.0
                        );
                    }
                }
            }
        }

        // === Phase 10: Stagnation tracking ===
        if hof_changed {
            stagnation_counter = 0;
        } else {
            stagnation_counter += 1;
        }

        log::info!(
            "Buffer: {}/{} | HoF: {}/{} | Stagnation: {}/{}",
            buffer.len(), config.buffer_capacity,
            hof.len(), config.hof_max_size,
            stagnation_counter, config.chaos_stagnation_threshold
        );

        // === Phase 11: Periodic checkpoint ===
        if (gen + 1) % config.checkpoint_interval == 0 {
            log::info!("💾 Saving checkpoint (gen {})...", gen + 1);
            save_model_checkpoint(&model);
            buffer.save(BUFFER_CHECKPOINT_PATH);
            hof.save(HOF_CHECKPOINT_PATH);
        }
    }

    // Final save
    log::info!("=== EVOLUTION COMPLETE ===");
    log::info!("Final save...");
    save_model_checkpoint(&model);
    buffer.save(BUFFER_CHECKPOINT_PATH);
    hof.save(HOF_CHECKPOINT_PATH);
    log::info!(
        "Done! Buffer: {} sequences, HoF: {} champions",
        buffer.len(),
        hof.len()
    );
}

// ============================================================================
// Helpers
// ============================================================================

/// Seed the buffer with random valid SUBLEQ programs
fn seed_random_programs(buffer: &mut ReplayBuffer, count: usize, max_tokens: usize) {
    for _ in 0..count {
        let (_, tokens) = generate_random_program(max_tokens);
        let ids = tokens_to_ids(&tokens);
        buffer.push(ids);
    }
}

/// Generate a random valid SUBLEQ program
fn generate_random_program(max_tokens: usize) -> (Vec<i64>, Vec<Token>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let num_instructions = rng.gen_range(1..=(max_tokens / 3).max(1));

    let mut tokens = vec![Token::Start];
    let mut program = Vec::new();

    for _ in 0..num_instructions {
        let a = rng.gen_range(0..constraint::NUM_ADDRESSES as u8);
        let b = rng.gen_range(0..constraint::NUM_ADDRESSES as u8);
        let c = rng.gen_range(0..constraint::NUM_ADDRESSES as u8);
        tokens.push(Token::Addr(a));
        tokens.push(Token::Addr(b));
        tokens.push(Token::Addr(c));
        program.push(a as i64);
        program.push(b as i64);
        program.push(c as i64);
    }

    tokens.push(Token::End);
    (program, tokens)
}
