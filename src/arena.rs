/// Battle Royale Arena
///
/// A shared-memory arena where multiple SUBLEQ gladiators fight to the death.
/// Programs are loaded at evenly-spaced offsets and execute round-robin.
/// They can read/write anywhere in shared memory, allowing them to corrupt
/// each other's code. Last survivor wins.

use crate::vm::SubleqVM;

/// Configuration for the arena
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Total shared memory size (default: 1024)
    pub memory_size: usize,
    /// Size of each gladiator's slot (default: 64)
    pub gladiator_slot_size: usize,
    /// Maximum total rounds before declaring a draw
    pub max_rounds: u32,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        ArenaConfig {
            memory_size: 1024,
            gladiator_slot_size: 64,
            max_rounds: 100_000,
        }
    }
}

/// Result of a battle
#[derive(Debug, Clone)]
pub struct BattleResult {
    /// Index of the winning gladiator (None if all died)
    pub winner_index: Option<usize>,
    /// Total rounds executed
    pub total_rounds: u32,
    /// Number of gladiators that survived
    pub survivors: usize,
    /// Order of elimination (first eliminated → first in list)
    pub elimination_order: Vec<usize>,
}

/// The Battle Royale Arena
pub struct Arena {
    /// Shared memory space
    pub memory: Vec<i64>,
    /// All gladiators
    pub gladiators: Vec<SubleqVM>,
    /// Arena configuration
    pub config: ArenaConfig,
}

impl Arena {
    /// Create a new empty arena
    pub fn new(config: ArenaConfig) -> Self {
        Arena {
            memory: vec![0; config.memory_size],
            gladiators: Vec::new(),
            config,
        }
    }

    /// Spawn gladiators from a list of programs.
    /// Each program is placed at `i * gladiator_slot_size` in shared memory.
    ///
    /// Panics if there are too many gladiators for the memory size.
    pub fn spawn(&mut self, programs: &[Vec<i64>]) {
        let max_gladiators = self.config.memory_size / self.config.gladiator_slot_size;
        assert!(
            programs.len() <= max_gladiators,
            "Too many gladiators ({}) for arena (max {})",
            programs.len(),
            max_gladiators
        );

        // Reset memory
        self.memory.fill(0);
        self.gladiators.clear();

        for (i, program) in programs.iter().enumerate() {
            let base_addr = i * self.config.gladiator_slot_size;
            let vm = SubleqVM::new(program, base_addr, &mut self.memory);
            self.gladiators.push(vm);
        }
    }

    /// Run the battle royale to completion.
    ///
    /// Gladiators execute round-robin (each alive gladiator gets one step per round).
    /// A gladiator is eliminated when its `step()` returns false.
    /// Battle ends when 0 or 1 gladiator remains, or max_rounds is reached.
    pub fn run_battle(&mut self) -> BattleResult {
        let mut round = 0u32;
        let mut elimination_order = Vec::new();

        loop {
            let alive_count = self.gladiators.iter().filter(|g| g.alive).count();

            // End conditions
            if alive_count <= 1 || round >= self.config.max_rounds {
                break;
            }

            // Round-robin: each alive gladiator gets one step
            for i in 0..self.gladiators.len() {
                if self.gladiators[i].alive {
                    let survived = self.gladiators[i].step(&mut self.memory);
                    if !survived {
                        elimination_order.push(i);
                    }
                }
            }

            round += 1;
        }

        let survivors = self.gladiators.iter().filter(|g| g.alive).count();
        let winner_index = if survivors == 1 {
            self.gladiators.iter().position(|g| g.alive)
        } else {
            None
        };

        BattleResult {
            winner_index,
            total_rounds: round,
            survivors,
            elimination_order,
        }
    }

    /// Extract the program (memory slice) of a gladiator by index.
    /// Returns the gladiator's slot from shared memory.
    pub fn extract_program(&self, index: usize) -> Vec<i64> {
        let base = index * self.config.gladiator_slot_size;
        let end = (base + self.config.gladiator_slot_size).min(self.config.memory_size);
        self.memory[base..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_placement() {
        let config = ArenaConfig {
            memory_size: 256,
            gladiator_slot_size: 64,
            max_rounds: 1000,
        };
        let mut arena = Arena::new(config);

        let prog_a = vec![1, 2, 3];
        let prog_b = vec![4, 5, 6];
        arena.spawn(&[prog_a, prog_b]);

        assert_eq!(arena.gladiators.len(), 2);
        assert_eq!(arena.gladiators[0].base_addr, 0);
        assert_eq!(arena.gladiators[1].base_addr, 64);

        // Check memory was loaded
        assert_eq!(arena.memory[0], 1);
        assert_eq!(arena.memory[1], 2);
        assert_eq!(arena.memory[2], 3);
        assert_eq!(arena.memory[64], 4);
        assert_eq!(arena.memory[65], 5);
        assert_eq!(arena.memory[66], 6);
    }

    #[test]
    fn test_battle_one_survivor() {
        let config = ArenaConfig {
            memory_size: 256,
            gladiator_slot_size: 64,
            max_rounds: 10_000,
        };
        let mut arena = Arena::new(config);

        // Gladiator 0: immediately halts (invalid address)
        let prog_halt = vec![999, 0, 0]; // A=999 is out of bounds → immediate death

        // Gladiator 1: self-loop (will survive until cycle limit)
        // A=64, B=64, C=64 → mem[64] -= mem[64] = 0, <=0 so jump to 64 (loop)
        let prog_loop = vec![64, 64, 64];

        arena.spawn(&[prog_halt, prog_loop]);
        let result = arena.run_battle();

        assert_eq!(result.winner_index, Some(1));
        assert_eq!(result.survivors, 1);
        assert!(result.elimination_order.contains(&0));
    }

    #[test]
    fn test_battle_all_die() {
        let config = ArenaConfig {
            memory_size: 256,
            gladiator_slot_size: 64,
            max_rounds: 10_000,
        };
        let mut arena = Arena::new(config);

        // Both programs immediately halt (out-of-bounds addresses)
        let prog_a = vec![999, 0, 0];
        let prog_b = vec![999, 0, 0];

        arena.spawn(&[prog_a, prog_b]);
        let result = arena.run_battle();

        assert_eq!(result.winner_index, None);
        assert_eq!(result.survivors, 0);
    }

    #[test]
    fn test_extract_program() {
        let config = ArenaConfig {
            memory_size: 128,
            gladiator_slot_size: 64,
            max_rounds: 1000,
        };
        let mut arena = Arena::new(config);

        let prog = vec![10, 20, 30];
        arena.spawn(&[prog]);

        let extracted = arena.extract_program(0);
        assert_eq!(extracted.len(), 64);
        assert_eq!(extracted[0], 10);
        assert_eq!(extracted[1], 20);
        assert_eq!(extracted[2], 30);
        assert_eq!(extracted[3], 0); // Rest is zero-initialized
    }
}
