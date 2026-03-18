/// SUBLEQ Transformer Model
///
/// A small autoregressive Transformer built on the Burn framework.
/// It generates sequences of SUBLEQ tokens, with logit masking
/// enforced by the SubleqConstraint state machine.

use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
};

use crate::constraint::{
    SubleqConstraint, Token, VOCAB_SIZE, END_TOKEN,
};

/// Configuration for the SUBLEQ Transformer
#[derive(Config, Debug)]
pub struct SubleqTransformerConfig {
    /// Model dimension
    #[config(default = 128)]
    pub d_model: usize,
    /// Number of attention heads
    #[config(default = 4)]
    pub n_heads: usize,
    /// Number of transformer layers
    #[config(default = 3)]
    pub n_layers: usize,
    /// Maximum sequence length
    #[config(default = 64)]
    pub max_seq_length: usize,
    /// Dropout rate
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// The SUBLEQ Transformer model
#[derive(Module, Debug)]
pub struct SubleqTransformer<B: Backend> {
    /// Token embeddings
    embedding_token: Embedding<B>,
    /// Positional embeddings
    embedding_pos: Embedding<B>,
    /// Transformer encoder (used autoregressively with causal mask)
    transformer: TransformerEncoder<B>,
    /// Output projection to vocabulary logits
    output: Linear<B>,
    /// Maximum sequence length
    max_seq_length: usize,
}

impl SubleqTransformerConfig {
    /// Initialize the model on the given device
    pub fn init<B: Backend>(&self, device: &B::Device) -> SubleqTransformer<B> {
        let transformer_config = TransformerEncoderConfig::new(self.d_model, self.d_model * 4, self.n_heads, self.n_layers)
            .with_dropout(self.dropout);

        let transformer = transformer_config.init(device);
        let embedding_token = EmbeddingConfig::new(VOCAB_SIZE, self.d_model).init(device);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.d_model).init(device);
        let output = LinearConfig::new(self.d_model, VOCAB_SIZE).init(device);

        SubleqTransformer {
            embedding_token,
            embedding_pos,
            transformer,
            output,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> SubleqTransformer<B> {
    /// Forward pass: tokens [batch, seq_len] → logits [batch, seq_len, vocab_size]
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_length] = tokens.dims();
        let device = &tokens.device();

        // Position indices [0, 1, 2, ..., seq_length-1]
        let index_positions = Tensor::<B, 1, Int>::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        // Embeddings
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = embedding_positions + embedding_tokens;

        // Autoregressive (causal) mask — prevents attending to future tokens
        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);

        // Transformer forward
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding).mask_attn(mask_attn),
        );

        // Project to vocab logits
        self.output.forward(encoded)
    }

    /// Compute training loss with teacher forcing.
    ///
    /// `tokens_input`: [batch, seq_len] — input tokens (shifted right)
    /// `targets`: [batch, seq_len] — target tokens (shifted left)
    ///
    /// Returns scalar loss tensor.
    pub fn forward_training(
        &self,
        tokens_input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_length] = tokens_input.dims();

        let logits = self.forward(tokens_input);

        let logits_flat = logits.reshape([batch_size * seq_length, VOCAB_SIZE]);
        let targets_flat = targets.reshape([batch_size * seq_length]);

        let loss_fn = CrossEntropyLossConfig::new()
            .init(&logits_flat.device());

        loss_fn.forward(logits_flat, targets_flat)
    }

    /// Generate a single SUBLEQ program using constrained decoding.
    ///
    /// Uses the constraint state machine to mask invalid tokens at each step.
    /// `temperature`: controls randomness (higher = more random, lower = more greedy)
    ///
    /// Returns the generated token sequence (including START and END).
    pub fn generate(
        &self,
        device: &B::Device,
        temperature: f64,
        max_len: usize,
    ) -> Vec<Token> {
        let max_len = max_len.min(self.max_seq_length);
        let mut constraint = SubleqConstraint::new();
        let mut generated_ids: Vec<i64> = Vec::new();
        let mut generated_tokens: Vec<Token> = Vec::new();

        // Start with START token
        let start_token = Token::Start;
        generated_ids.push(start_token.to_id() as i64);
        generated_tokens.push(start_token);
        constraint.advance(start_token);

        for _ in 1..max_len {
            if constraint.is_done() {
                break;
            }

            // Build input tensor [1, current_seq_len]
            let seq_len = generated_ids.len();
            let input_data: Vec<i64> = generated_ids.clone();
            let input_tensor = Tensor::<B, 1, Int>::from_data(
                &input_data[..],
                device,
            )
            .reshape([1, seq_len]);

            // Forward pass → logits [1, seq_len, vocab_size]
            let logits = self.forward(input_tensor);

            // Take logits for the last position [1, vocab_size]
            let last_logits = logits
                .slice([0..1, (seq_len - 1)..seq_len, 0..VOCAB_SIZE])
                .reshape([VOCAB_SIZE]);

            // Apply constraint mask
            let mask = constraint.allowed_token_mask();
            let mask_tensor = Tensor::<B, 1>::from_data(
                &mask.iter().map(|&b| if b { 0.0f32 } else { f32::NEG_INFINITY }).collect::<Vec<_>>()[..],
                device,
            );
            let masked_logits = last_logits + mask_tensor;

            // Apply temperature and softmax
            let scaled = masked_logits / temperature;
            let probs = burn::tensor::activation::softmax(scaled.reshape([1, VOCAB_SIZE]), 1)
                .reshape([VOCAB_SIZE]);

            // Sample from distribution (argmax for now — simple greedy/near-greedy)
            let probs_data: Vec<f32> = probs.to_data().to_vec().unwrap();
            let sampled_id = sample_from_probs(&probs_data);

            let token = Token::from_id(sampled_id).expect("Invalid token ID sampled");
            generated_ids.push(sampled_id as i64);
            generated_tokens.push(token);
            constraint.advance(token);
        }

        // If we hit max_len without END, force an END if in valid state
        if !constraint.is_done() {
            if constraint.state == crate::constraint::GenState::ExpectEndOrA {
                generated_tokens.push(Token::End);
            }
        }

        generated_tokens
    }
}

/// Sample an index from a probability distribution using a simple
/// weighted random selection. Falls back to argmax if rand is not seeded well.
fn sample_from_probs(probs: &[f32]) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    // Fallback: return the last valid index (shouldn't normally reach here)
    probs.len() - 1
}

/// Utility: Compute gradients and return loss value for a training step.
/// This is used by the training loop.
pub fn training_step<B: AutodiffBackend>(
    model: &SubleqTransformer<B>,
    tokens_input: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
) -> (Tensor<B, 1>, B::Gradients) {
    let loss = model.forward_training(tokens_input, targets);
    let grads = loss.backward();
    (loss, grads)
}
