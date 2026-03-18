/// SUBLEQ Virtual Machine
///
/// Executes SUBLEQ (SUBtract and branch if Less-than or EQual to zero) instructions
/// on a shared memory space. Each instruction is a triplet (A, B, C):
///   mem[B] -= mem[A]
///   if mem[B] <= 0 then PC = C else PC += 3

/// Maximum number of cycles before forced halt (prevents infinite loops)
pub const MAX_CYCLES: u32 = 10_000;

/// A single SUBLEQ gladiator executing on shared memory
#[derive(Debug, Clone)]
pub struct SubleqVM {
    /// Program counter (index into shared memory)
    pub pc: usize,
    /// Number of cycles executed
    pub cycles: u32,
    /// Whether this VM is still alive
    pub alive: bool,
    /// Base address where this gladiator was spawned (for identification)
    pub base_addr: usize,
    /// Size of the original program loaded
    pub program_len: usize,
}

impl SubleqVM {
    /// Create a new VM and load the program into shared memory at `base_addr`.
    ///
    /// Returns the VM with PC set to `base_addr`.
    pub fn new(program: &[i64], base_addr: usize, memory: &mut [i64]) -> Self {
        let mem_size = memory.len();
        let mut loaded = 0;
        for (i, &val) in program.iter().enumerate() {
            let addr = base_addr + i;
            if addr < mem_size {
                memory[addr] = val;
                loaded += 1;
            }
        }

        SubleqVM {
            pc: base_addr,
            cycles: 0,
            alive: true,
            base_addr,
            program_len: loaded,
        }
    }

    /// Execute one SUBLEQ instruction on the given shared memory.
    ///
    /// Returns `true` if the VM is still alive after this step.
    /// Returns `false` if the VM has halted (and sets `self.alive = false`).
    pub fn step(&mut self, memory: &mut [i64]) -> bool {
        if !self.alive {
            return false;
        }

        let mem_size = memory.len();

        // Check if we can read the 3 operands
        if self.pc + 2 >= mem_size {
            self.alive = false;
            return false;
        }

        // Check cycle limit
        if self.cycles >= MAX_CYCLES {
            self.alive = false;
            return false;
        }

        // Read operands A, B, C
        let a = memory[self.pc] as usize;
        let b = memory[self.pc + 1] as usize;
        let c = memory[self.pc + 2] as usize;

        // Validate A and B are within bounds
        if a >= mem_size || b >= mem_size {
            self.alive = false;
            return false;
        }

        // Core SUBLEQ operation: mem[B] -= mem[A]
        memory[b] = memory[b].wrapping_sub(memory[a]);

        // Branch: if mem[B] <= 0, jump to C; otherwise advance PC by 3
        if memory[b] <= 0 {
            if c >= mem_size {
                // C points out of bounds → halt
                self.alive = false;
                return false;
            }
            self.pc = c;
        } else {
            self.pc += 3;
        }

        self.cycles += 1;
        true
    }

    /// Run until the VM halts (step returns false).
    /// Useful for standalone execution (not battle royale mode).
    pub fn run_to_death(&mut self, memory: &mut [i64]) {
        while self.step(memory) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_subtraction() {
        // Program: mem[1] -= mem[0], then halt (PC goes out of bounds)
        // Memory layout: [3, 7, 99, ...]  (99 = jump target out of range to halt after)
        // After step: mem[1] = 7 - 3 = 4 (positive, so PC += 3 → out of bounds → halt next step)
        let mut memory = [0i64; 64];
        let program = vec![0, 1, 99]; // A=0, B=1, C=99
        memory[0] = 3; // Will be overwritten by program load...
        let mut vm = SubleqVM::new(&program, 0, &mut memory);
        // After loading: mem = [0, 1, 99, ...] — those are address values
        // We need actual values at the addressed locations
        // A=0 → mem[0]=0, B=1 → mem[1]=1, so mem[1] -= mem[0] → 1 - 0 = 1
        assert!(vm.step(&mut memory)); // PC goes to 3 (positive result)
        assert_eq!(memory[1], 1); // 1 - 0 = 1
    }

    #[test]
    fn test_branch_on_negative() {
        // Set up: mem[0]=5, mem[1]=3, instruction at [2,3,4] = A=0, B=1, C=10
        // mem[1] -= mem[0] → 3 - 5 = -2 (<=0), so jump to C=10
        let mut memory = [0i64; 64];
        memory[0] = 5;
        memory[1] = 3;
        memory[2] = 0; // A
        memory[3] = 1; // B
        memory[4] = 10; // C (jump target)

        let mut vm = SubleqVM {
            pc: 2,
            cycles: 0,
            alive: true,
            base_addr: 2,
            program_len: 3,
        };

        assert!(vm.step(&mut memory));
        assert_eq!(memory[1], -2); // 3 - 5 = -2
        assert_eq!(vm.pc, 10); // Jumped to C
    }

    #[test]
    fn test_halt_on_out_of_bounds_pc() {
        let mut memory = [0i64; 8];
        let mut vm = SubleqVM {
            pc: 6, // pc + 2 = 8 >= mem_size(8)
            cycles: 0,
            alive: true,
            base_addr: 0,
            program_len: 0,
        };
        assert!(!vm.step(&mut memory));
        assert!(!vm.alive);
    }

    #[test]
    fn test_halt_on_cycle_limit() {
        // A self-looping program: A=0, B=0, C=0
        // mem[0] -= mem[0] → always 0, which is <=0, so jump to 0 (infinite loop)
        // But should halt at MAX_CYCLES
        let mut memory = [0i64; 64];
        memory[0] = 0;
        memory[1] = 0;
        memory[2] = 0;

        let mut vm = SubleqVM::new(&[0, 0, 0], 0, &mut memory);
        vm.run_to_death(&mut memory);
        assert!(!vm.alive);
        assert_eq!(vm.cycles, MAX_CYCLES);
    }

    #[test]
    fn test_halt_on_invalid_address() {
        // A points to address 999 which is out of bounds
        let mut memory = [0i64; 64];
        memory[0] = 999; // A
        memory[1] = 0;   // B
        memory[2] = 0;   // C

        let mut vm = SubleqVM::new(&[999, 0, 0], 0, &mut memory);
        assert!(!vm.step(&mut memory));
        assert!(!vm.alive);
    }
}
