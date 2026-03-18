/// Training Loop & Replay Buffer
///
/// Implements the self-play evolutionary loop:
/// 1. Generate N programs via constrained Transformer
/// 2. Run Battle Royale in the Arena
/// 3. Store winner's token sequence in the Replay Buffer
/// 4. Train the Transformer on buffered winning sequences
/// 5. Repeat for G generations

use burn::prelude::*;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;

use crate::arena::{Arena, ArenaConfig, BattleResult};
use crate::constraint::{
    self, decode_tokens, encode_program, tokens_to_ids, Token, VOCAB_SIZE, END_TOKEN,
};
use crate::transformer::{SubleqTransformer, SubleqTransformerConfig, training_step};

/// Replay buffer storing winning token sequences
#[derive(Debug)]
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

    /// Create a training batch from the buffer.
    ///
    /// Teacher forcing: input = tokens[:-1], target = tokens[1:]
    /// Sequences are padded to the maximum length in the batch.
    pub fn make_batch<B: Backend>(
        &self,
        device: &B::Device,
        max_seq_len: usize,
    ) -> Option<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
        if self.sequences.is_empty() {
            return None;
        }

        let batch_size = self.sequences.len();
        // Clamp sequence length
        let actual_max = self.sequences.iter().map(|s| s.len()).max().unwrap_or(1);
        let seq_len = actual_max.min(max_seq_len);

        // Build padded input and target arrays
        let mut input_data = vec![0i64; batch_size * (seq_len - 1)];
        let mut target_data = vec![0i64; batch_size * (seq_len - 1)];

        for (i, seq) in self.sequences.iter().enumerate() {
            for j in 0..(seq_len - 1) {
                let input_idx = i * (seq_len - 1) + j;
                if j < seq.len() {
                    input_data[input_idx] = seq[j] as i64;
                }
                if j + 1 < seq.len() {
                    target_data[input_idx] = seq[j + 1] as i64;
                } else {
                    target_data[input_idx] = END_TOKEN as i64; // Pad targets with END
                }
            }
        }

        let input_tensor = Tensor::<B, 1, Int>::from_data(&input_data[..], device)
            .reshape([batch_size, seq_len - 1]);
        let target_tensor = Tensor::<B, 1, Int>::from_data(&target_data[..], device)
            .reshape([batch_size, seq_len - 1]);

        Some((input_tensor, target_tensor))
    }
}

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
            buffer_capacity: 500,
            arena_config: ArenaConfig::default(),
            transformer_config: SubleqTransformerConfig::new(),
        }
    }
}

/// Run the full evolution loop.
///
/// This is the main "cỗ máy cày" — the grinding engine that:
/// 1. Generates a population of SUBLEQ programs
/// 2. Throws them into the arena
/// 3. Keeps the winner's DNA
/// 4. Trains the Transformer to produce more programs like the winner
/// 5. Repeats until convergence or generation limit
pub fn evolution_loop<B: AutodiffBackend>(
    config: EvolutionConfig,
    device: &B::Device,
) {
    log::info!("=== THE SUBLEQ SPIRE ===");
    log::info!("Initializing Transformer model...");

    // Initialize model
    let model = config.transformer_config.init::<B>(device);
    let mut model = model;

    // Initialize optimizer
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init::<B, SubleqTransformer<B>>();

    // Initialize replay buffer
    let mut buffer = ReplayBuffer::new(config.buffer_capacity);

    // Seed the buffer with some random programs to bootstrap training
    log::info!("Seeding replay buffer with random programs...");
    seed_random_programs(&mut buffer, 20, config.max_program_tokens);

    log::info!("Starting evolution for {} generations...", config.num_generations);

    for gen in 0..config.num_generations {
        log::info!("--- Generation {} ---", gen + 1);

        // === Phase 1: Generate population ===
        let mut programs: Vec<Vec<i64>> = Vec::new();
        let mut token_sequences: Vec<Vec<Token>> = Vec::new();

        for _ in 0..config.gladiators_per_battle {
            let tokens = model.generate(device, config.temperature, config.max_program_tokens);
            let program = decode_tokens(&tokens);
            if program.len() >= 3 {
                // Only accept valid programs (at least 1 instruction)
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
            "Battle finished in {} rounds. Survivors: {}",
            result.total_rounds,
            result.survivors
        );

        // === Phase 3: Collect winner ===
        if let Some(winner_idx) = result.winner_index {
            log::info!(
                "Winner: Gladiator {} (survived {} cycles)",
                winner_idx,
                arena.gladiators[winner_idx].cycles
            );

            // Store winner's token sequence in replay buffer
            let winner_ids = tokens_to_ids(&token_sequences[winner_idx]);
            buffer.push(winner_ids);
        } else {
            log::info!("No winner — all gladiators eliminated.");
            // Use the last eliminated as the "least bad"
            if let Some(&last_eliminated) = result.elimination_order.last() {
                let ids = tokens_to_ids(&token_sequences[last_eliminated]);
                buffer.push(ids);
                log::info!("Using last eliminated (Gladiator {}) as surrogate winner.", last_eliminated);
            }
        }

        // === Phase 4: Train on replay buffer ===
        if buffer.len() >= 2 {
            for step in 0..config.train_steps_per_gen {
                if let Some((input, target)) = buffer.make_batch::<B>(device, config.max_program_tokens) {
                    let (loss, grads) = training_step(&model, input, target);

                    // Extract loss value for logging
                    let loss_val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];

                    // Apply gradients
                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(config.learning_rate, model, grads);

                    if step == 0 {
                        log::info!("Training loss: {:.4}", loss_val);
                    }
                }
            }
        }

        log::info!("Replay buffer size: {}", buffer.len());
    }

    log::info!("=== EVOLUTION COMPLETE ===");
    log::info!("Final buffer size: {}", buffer.len());
}

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

    // Random number of instructions (1 to max_tokens/3)
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
