/// Constrained State Machine for SUBLEQ Token Generation
///
/// Enforces that the Transformer can only generate syntactically valid
/// SUBLEQ programs: sequences of triplets (A, B, C) where each element
/// is an address in [0, 63], bookended by START and END tokens.

/// Number of memory addresses in the SUBLEQ arena per gladiator
pub const NUM_ADDRESSES: usize = 64;

/// Total vocabulary size: START + END + 64 addresses = 66
pub const VOCAB_SIZE: usize = NUM_ADDRESSES + 2;

/// Special token IDs
pub const START_TOKEN: usize = 0;
pub const END_TOKEN: usize = 1;
pub const ADDR_OFFSET: usize = 2;

/// A token in the SUBLEQ vocabulary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Start,
    End,
    Addr(u8), // 0..63
}

impl Token {
    /// Convert token to its numeric ID
    pub fn to_id(self) -> usize {
        match self {
            Token::Start => START_TOKEN,
            Token::End => END_TOKEN,
            Token::Addr(a) => ADDR_OFFSET + a as usize,
        }
    }

    /// Convert numeric ID to token
    pub fn from_id(id: usize) -> Option<Token> {
        if id == START_TOKEN {
            Some(Token::Start)
        } else if id == END_TOKEN {
            Some(Token::End)
        } else if id >= ADDR_OFFSET && id < VOCAB_SIZE {
            Some(Token::Addr((id - ADDR_OFFSET) as u8))
        } else {
            None
        }
    }
}

/// States of the constrained generation FSM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenState {
    /// Expecting the START token
    ExpectStart,
    /// Expecting operand A (first of a triplet)
    ExpectA,
    /// Expecting operand B (second of a triplet)
    ExpectB,
    /// Expecting operand C (third of a triplet)
    ExpectC,
    /// Expecting either END or a new operand A
    ExpectEndOrA,
    /// Generation is complete
    Done,
}

/// The constrained decoder state machine
#[derive(Debug, Clone)]
pub struct SubleqConstraint {
    pub state: GenState,
}

impl SubleqConstraint {
    pub fn new() -> Self {
        SubleqConstraint {
            state: GenState::ExpectStart,
        }
    }

    /// Returns a boolean mask of size VOCAB_SIZE where `true` means the token is allowed.
    /// This mask is used to filter logits before sampling.
    pub fn allowed_token_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; VOCAB_SIZE];

        match self.state {
            GenState::ExpectStart => {
                mask[START_TOKEN] = true;
            }
            GenState::ExpectA | GenState::ExpectB | GenState::ExpectC => {
                // Only address tokens allowed
                for i in 0..NUM_ADDRESSES {
                    mask[ADDR_OFFSET + i] = true;
                }
            }
            GenState::ExpectEndOrA => {
                // END or any address token
                mask[END_TOKEN] = true;
                for i in 0..NUM_ADDRESSES {
                    mask[ADDR_OFFSET + i] = true;
                }
            }
            GenState::Done => {
                // Nothing is allowed, generation is over
            }
        }

        mask
    }

    /// Advance the state machine after a token has been selected.
    ///
    /// Panics if the token is not allowed in the current state
    /// (should never happen if `allowed_token_mask` is respected).
    pub fn advance(&mut self, token: Token) {
        self.state = match self.state {
            GenState::ExpectStart => {
                assert_eq!(token, Token::Start, "Expected START token");
                GenState::ExpectA
            }
            GenState::ExpectA => {
                assert!(matches!(token, Token::Addr(_)), "Expected address token for A");
                GenState::ExpectB
            }
            GenState::ExpectB => {
                assert!(matches!(token, Token::Addr(_)), "Expected address token for B");
                GenState::ExpectC
            }
            GenState::ExpectC => {
                assert!(matches!(token, Token::Addr(_)), "Expected address token for C");
                GenState::ExpectEndOrA
            }
            GenState::ExpectEndOrA => match token {
                Token::End => GenState::Done,
                Token::Addr(_) => GenState::ExpectB, // This was A of a new triplet
                Token::Start => panic!("START not allowed here"),
            },
            GenState::Done => panic!("Generation already complete"),
        };
    }

    /// Check if generation is complete
    pub fn is_done(&self) -> bool {
        self.state == GenState::Done
    }
}

/// Decode a sequence of tokens into a SUBLEQ program (list of i64 addresses).
///
/// Strips START and END tokens, converts Addr tokens to their numeric values.
/// The resulting Vec should have length divisible by 3 (each triplet = one instruction).
pub fn decode_tokens(tokens: &[Token]) -> Vec<i64> {
    tokens
        .iter()
        .filter_map(|t| match t {
            Token::Addr(a) => Some(*a as i64),
            _ => None, // Skip START and END
        })
        .collect()
}

/// Encode a SUBLEQ program (list of i64 addresses) into a token sequence.
///
/// Wraps with START and END tokens.
pub fn encode_program(program: &[i64]) -> Vec<Token> {
    let mut tokens = vec![Token::Start];
    for &addr in program {
        let a = addr.clamp(0, (NUM_ADDRESSES - 1) as i64) as u8;
        tokens.push(Token::Addr(a));
    }
    tokens.push(Token::End);
    tokens
}

/// Convert a token sequence to a list of token IDs (for model input)
pub fn tokens_to_ids(tokens: &[Token]) -> Vec<usize> {
    tokens.iter().map(|t| t.to_id()).collect()
}

/// Convert a list of token IDs back to tokens
pub fn ids_to_tokens(ids: &[usize]) -> Vec<Token> {
    ids.iter().filter_map(|&id| Token::from_id(id)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_roundtrip() {
        for id in 0..VOCAB_SIZE {
            let token = Token::from_id(id).unwrap();
            assert_eq!(token.to_id(), id);
        }
    }

    #[test]
    fn test_start_only_allows_start() {
        let c = SubleqConstraint::new();
        let mask = c.allowed_token_mask();
        assert!(mask[START_TOKEN]);
        assert!(!mask[END_TOKEN]);
        for i in ADDR_OFFSET..VOCAB_SIZE {
            assert!(!mask[i]);
        }
    }

    #[test]
    fn test_valid_sequence() {
        let mut c = SubleqConstraint::new();

        // START
        c.advance(Token::Start);
        assert_eq!(c.state, GenState::ExpectA);

        // First triplet: A=10, B=20, C=30
        c.advance(Token::Addr(10));
        assert_eq!(c.state, GenState::ExpectB);
        c.advance(Token::Addr(20));
        assert_eq!(c.state, GenState::ExpectC);
        c.advance(Token::Addr(30));
        assert_eq!(c.state, GenState::ExpectEndOrA);

        // Second triplet: A=5, B=5, C=0
        c.advance(Token::Addr(5));
        assert_eq!(c.state, GenState::ExpectB);
        c.advance(Token::Addr(5));
        assert_eq!(c.state, GenState::ExpectC);
        c.advance(Token::Addr(0));
        assert_eq!(c.state, GenState::ExpectEndOrA);

        // END
        c.advance(Token::End);
        assert!(c.is_done());
    }

    #[test]
    fn test_expect_end_or_a_mask() {
        let mut c = SubleqConstraint::new();
        c.advance(Token::Start);
        c.advance(Token::Addr(0));
        c.advance(Token::Addr(0));
        c.advance(Token::Addr(0));
        // Now in ExpectEndOrA
        let mask = c.allowed_token_mask();
        assert!(!mask[START_TOKEN]);
        assert!(mask[END_TOKEN]);
        for i in ADDR_OFFSET..VOCAB_SIZE {
            assert!(mask[i]);
        }
    }

    #[test]
    fn test_decode_tokens() {
        let tokens = vec![
            Token::Start,
            Token::Addr(10),
            Token::Addr(15),
            Token::Addr(4),
            Token::Addr(2),
            Token::Addr(2),
            Token::Addr(8),
            Token::End,
        ];
        let program = decode_tokens(&tokens);
        assert_eq!(program, vec![10, 15, 4, 2, 2, 8]);
    }

    #[test]
    fn test_encode_program() {
        let program = vec![10, 15, 4];
        let tokens = encode_program(&program);
        assert_eq!(tokens.len(), 5); // START + 3 addrs + END
        assert_eq!(tokens[0], Token::Start);
        assert_eq!(tokens[4], Token::End);
    }

    #[test]
    fn test_ids_roundtrip() {
        let tokens = vec![Token::Start, Token::Addr(5), Token::Addr(10), Token::Addr(63), Token::End];
        let ids = tokens_to_ids(&tokens);
        let back = ids_to_tokens(&ids);
        assert_eq!(tokens, back);
    }

    #[test]
    #[should_panic(expected = "Expected START token")]
    fn test_invalid_start() {
        let mut c = SubleqConstraint::new();
        c.advance(Token::Addr(5)); // Should panic — expected START
    }
}
