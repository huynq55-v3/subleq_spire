/// Training Loop & Replay Buffer
///
/// Implements the self-play evolutionary loop:
/// 1. Generate N programs via constrained Transformer
/// 2. Run Battle Royale in the Arena
/// 3. Store winner's token sequence in the Replay Buffer
/// 4. Train the Transformer on buffered winning sequences (random subset)
/// 5. Save checkpoint & buffer to disk
/// 6. Repeat for G generations

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
use crate::transformer::{SubleqTransformer, SubleqTransformerConfig, training_step};

// ============================================================================
// Replay Buffer — with persistence and random batch sampling
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

    /// Get the number of sequences in the buffer
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if buffer is empty
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
                log::info!("Loaded replay buffer from {} ({} sequences)", path, buffer.sequences.len());
                return buffer;
            } else {
                log::warn!("Failed to parse replay buffer file, starting fresh");
            }
        }
        Self::new(default_capacity)
    }

    /// Create a training batch by sampling a random subset from the buffer.
    ///
    /// Teacher forcing: input = tokens[:-1], target = tokens[1:]
    /// Sequences are padded to the maximum length in the sampled batch.
    ///
    /// `batch_size`: number of sequences to sample (clamped to buffer size)
    pub fn make_batch<B: Backend>(
        &self,
        device: &B::Device,
        max_seq_len: usize,
        batch_size: usize,
    ) -> Option<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
        if self.sequences.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let actual_batch_size = batch_size.min(self.sequences.len());

        // Randomly sample `actual_batch_size` sequences
        let selected: Vec<&Vec<usize>> = self
            .sequences
            .choose_multiple(&mut rng, actual_batch_size)
            .collect();

        // Find max length in this sample
        let actual_max = selected.iter().map(|s| s.len()).max().unwrap_or(2);
        let seq_len = actual_max.min(max_seq_len).max(2); // at least 2 for input/target split

        // Build padded input and target arrays
        let target_len = seq_len - 1;
        let mut input_data = vec![0i64; actual_batch_size * target_len];
        let mut target_data = vec![0i64; actual_batch_size * target_len];

        for (i, seq) in selected.iter().enumerate() {
            for j in 0..target_len {
                let idx = i * target_len + j;
                if j < seq.len() {
                    input_data[idx] = seq[j] as i64;
                }
                if j + 1 < seq.len() {
                    target_data[idx] = seq[j + 1] as i64;
                } else {
                    target_data[idx] = END_TOKEN as i64; // Pad targets with END
                }
            }
        }

        let input_tensor = Tensor::<B, 1, Int>::from_data(&input_data[..], device)
            .reshape([actual_batch_size, target_len]);
        let target_tensor = Tensor::<B, 1, Int>::from_data(&target_data[..], device)
            .reshape([actual_batch_size, target_len]);

        Some((input_tensor, target_tensor))
    }
}

// ============================================================================
// Checkpoint management
// ============================================================================

const MODEL_CHECKPOINT_PATH: &str = "checkpoints/model";
const BUFFER_CHECKPOINT_PATH: &str = "checkpoints/replay_buffer.json";

/// Save model checkpoint to disk
fn save_model_checkpoint<B: Backend>(model: &SubleqTransformer<B>) {
    // Ensure checkpoints directory exists
    let _ = std::fs::create_dir_all("checkpoints");

    match model
        .clone()
        .save_file(MODEL_CHECKPOINT_PATH, &CompactRecorder::new())
    {
        Ok(_) => log::info!("Model checkpoint saved"),
        Err(e) => log::error!("Failed to save model checkpoint: {}", e),
    }
}

/// Try to load model checkpoint from disk. Returns None if no checkpoint exists.
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
    /// Number of gladiators per battle
    pub gladiators_per_battle: usize,
    /// Number of generations to run
    pub num_generations: usize,
    /// Number of training steps per generation
    pub train_steps_per_gen: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Temperature for token sampling
    pub temperature: f64,
    /// Maximum program length (in tokens)
    pub max_program_tokens: usize,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Training batch size (random subset of buffer per step)
    pub train_batch_size: usize,
    /// How often to save checkpoints (every N generations)
    pub checkpoint_interval: usize,
    /// Arena config
    pub arena_config: ArenaConfig,
    /// Transformer config
    pub transformer_config: SubleqTransformerConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        EvolutionConfig {
            gladiators_per_battle: 10,
            num_generations: 100,
            train_steps_per_gen: 5,
            learning_rate: 1e-3,
            temperature: 1.0,
            max_program_tokens: 64,
            buffer_capacity: 1000,
            train_batch_size: 32,
            checkpoint_interval: 10,
            arena_config: ArenaConfig::default(),
            transformer_config: SubleqTransformerConfig::new(),
        }
    }
}

// ============================================================================
// Evolution Loop
// ============================================================================

/// Run the full evolution loop.
///
/// This is the main "cỗ máy cày" — the grinding engine that:
/// 1. Loads checkpoint + buffer from disk (if available)
/// 2. Generates a population of SUBLEQ programs
/// 3. Throws them into the arena
/// 4. Keeps the winner's DNA
/// 5. Trains the Transformer on a random subset of winners
/// 6. Saves checkpoint & buffer periodically
/// 7. Repeats until convergence or generation limit
pub fn evolution_loop<B: AutodiffBackend>(
    config: EvolutionConfig,
    device: &B::Device,
) {
    log::info!("=== THE SUBLEQ SPIRE ===");

    // --- Load or initialize model ---
    let mut model = if let Some(loaded) = load_model_checkpoint::<B>(&config.transformer_config, device) {
        log::info!("Resuming from checkpoint!");
        loaded
    } else {
        log::info!("Initializing fresh Transformer model...");
        config.transformer_config.init::<B>(device)
    };

    // --- Initialize optimizer (fresh — Adam state is not checkpointed) ---
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init::<B, SubleqTransformer<B>>();

    // --- Load or initialize replay buffer ---
    let mut buffer = ReplayBuffer::load(BUFFER_CHECKPOINT_PATH, config.buffer_capacity);

    // Seed buffer if it's empty (first run)
    if buffer.is_empty() {
        log::info!("Seeding replay buffer with random programs...");
        seed_random_programs(&mut buffer, 50, config.max_program_tokens);
    }

    log::info!(
        "Config: {} gladiators, {} gens, batch_size={}, buffer_cap={}",
        config.gladiators_per_battle,
        config.num_generations,
        config.train_batch_size,
        config.buffer_capacity
    );
    log::info!("Starting evolution...");

    for gen in 0..config.num_generations {
        log::info!("--- Generation {} ---", gen + 1);

        // === Phase 1: Generate population ===
        let mut programs: Vec<Vec<i64>> = Vec::new();
        let mut token_sequences: Vec<Vec<Token>> = Vec::new();

        for _ in 0..config.gladiators_per_battle {
            let tokens = model.generate(device, config.temperature, config.max_program_tokens);
            let program = decode_tokens(&tokens);
            if program.len() >= 3 {
                programs.push(program);
                token_sequences.push(tokens);
            }
        }

        // Fill up with random programs if we don't have enough
        while programs.len() < config.gladiators_per_battle {
            let (prog, tokens) = generate_random_program(config.max_program_tokens);
            programs.push(prog);
            token_sequences.push(tokens);
        }

        log::info!("Generated {} gladiators", programs.len());

        // === Phase 2: Battle Royale ===
        let mut arena = Arena::new(config.arena_config.clone());
        arena.spawn(&programs);
        let result = arena.run_battle();

        log::info!(
            "Battle: {} rounds, {} survivors",
            result.total_rounds,
            result.survivors
        );

        // === Phase 3: Collect winner ===
        if let Some(winner_idx) = result.winner_index {
            log::info!(
                "🏆 Winner: Gladiator {} ({} cycles)",
                winner_idx,
                arena.gladiators[winner_idx].cycles
            );
            let winner_ids = tokens_to_ids(&token_sequences[winner_idx]);
            buffer.push(winner_ids);
        } else {
            log::info!("💀 No winner — all gladiators eliminated.");
            if let Some(&last_eliminated) = result.elimination_order.last() {
                let ids = tokens_to_ids(&token_sequences[last_eliminated]);
                buffer.push(ids);
                log::info!("Using last eliminated (Gladiator {}) as surrogate.", last_eliminated);
            }
        }

        // === Phase 4: Train on random subset of replay buffer ===
        if buffer.len() >= 2 {
            for step in 0..config.train_steps_per_gen {
                if let Some((input, target)) = buffer.make_batch::<B>(
                    device,
                    config.max_program_tokens,
                    config.train_batch_size,
                ) {
                    let (loss, grads) = training_step(&model, input, target);

                    let loss_val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];

                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(config.learning_rate, model, grads);

                    if step == 0 {
                        log::info!("Loss: {:.4} (batch={})", loss_val, config.train_batch_size);
                    }
                }
            }
        }

        log::info!("Buffer: {}/{}", buffer.len(), config.buffer_capacity);

        // === Phase 5: Periodic checkpoint ===
        if (gen + 1) % config.checkpoint_interval == 0 {
            log::info!("💾 Saving checkpoint (gen {})...", gen + 1);
            save_model_checkpoint(&model);
            buffer.save(BUFFER_CHECKPOINT_PATH);
        }
    }

    // Final save
    log::info!("=== EVOLUTION COMPLETE ===");
    log::info!("Final save...");
    save_model_checkpoint(&model);
    buffer.save(BUFFER_CHECKPOINT_PATH);
    log::info!("Done! Buffer: {} sequences", buffer.len());
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
