#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/vm/bytecode.h"
#include "../include/vm/vm_state.h"
#include "../include/vm/instructions.h"
#include "../include/utils/dump.h"
#include "../include/utils/timer.h"

static Instruction make_instruction(OpCode op, uint32_t operand) {
    Instruction instr;
    instr.opcode = op;
    instr.operand = operand;
    return instr;
}

BytecodeProgram* create_add_test() {
    BytecodeProgram* program = bytecode_create(5, 0);
    if (!program) return NULL;
    
    program->instructions[0] = make_instruction(OP_LOAD_IMM, (0 << 16) | 5);
    program->instructions[1] = make_instruction(OP_LOAD_IMM, (1 << 16) | 3);
    program->instructions[2] = make_instruction(OP_ADD, (0 << 16) | (0 << 8) | 1);
    program->instructions[3] = make_instruction(OP_STORE, (0 << 16) | 0);
    program->instructions[4] = make_instruction(OP_HALT, 0);
    
    return program;
}

BytecodeProgram* create_branch_test() {
    BytecodeProgram* program = bytecode_create(10, 0);
    if (!program) return NULL;
    
    program->instructions[0] = make_instruction(OP_LOAD_IMM, (0 << 16) | 10);
    program->instructions[1] = make_instruction(OP_LOAD_IMM, (1 << 16) | 5);
    program->instructions[2] = make_instruction(OP_SUB, (2 << 16) | (0 << 8) | 1);
    program->instructions[3] = make_instruction(OP_JMP_TRUE, (2 << 16) | 6);
    program->instructions[4] = make_instruction(OP_LOAD_IMM, (3 << 16) | 0);
    program->instructions[5] = make_instruction(OP_JMP, 8);
    program->instructions[6] = make_instruction(OP_LOAD_IMM, (3 << 16) | 1);
    program->instructions[7] = make_instruction(OP_STORE, (3 << 16) | 0);
    program->instructions[8] = make_instruction(OP_HALT, 0);
    program->instructions[9] = make_instruction(OP_NOP, 0);
    
    return program;
}

BytecodeProgram* create_loop_test() {
    BytecodeProgram* program = bytecode_create(15, 0);
    if (!program) return NULL;
    
    program->instructions[0] = make_instruction(OP_LOAD_IMM, (0 << 16) | 0);
    program->instructions[1] = make_instruction(OP_LOAD_IMM, (1 << 16) | 10);
    program->instructions[2] = make_instruction(OP_LOAD_IMM, (2 << 16) | 0);
    program->instructions[3] = make_instruction(OP_SUB, (3 << 16) | (0 << 8) | 1);
    program->instructions[4] = make_instruction(OP_JMP_FALSE, (3 << 16) | 9);
    program->instructions[5] = make_instruction(OP_ADD, (2 << 16) | (2 << 8) | 0);
    program->instructions[6] = make_instruction(OP_LOAD_IMM, (4 << 16) | 1);
    program->instructions[7] = make_instruction(OP_ADD, (0 << 16) | (0 << 8) | 4);
    program->instructions[8] = make_instruction(OP_JMP, 3);
    program->instructions[9] = make_instruction(OP_STORE, (2 << 16) | 0);
    program->instructions[10] = make_instruction(OP_HALT, 0);

    for (int i = 11; i < 15; i++) {
        program->instructions[i] = make_instruction(OP_NOP, 0);
    }
    
    return program;
}

int test_program_host(const char* test_name, BytecodeProgram* program) {
    printf("\n=== Test: %s ===\n", test_name);
    dump_program(program);
    
    VMState* state = vm_state_create(1024);
    if (!state) {
        printf("Error: Failed to create VM state\n");
        return -1;
    }
    
    int iterations = 0;
    int max_iterations = 10000;
    
    Timer timer;
    timer_start(&timer);
    
    while (state->pc < program->instruction_count && iterations < max_iterations) {
        Instruction* instr = &program->instructions[state->pc];
        
        int result = instruction_execute_host(
            instr,
            state->registers,
            state->stack,
            &state->stack_ptr,
            (int*)&state->pc,
            state->memory,
            state->memory_size
        );
        
        if (result > 0) {
            break;  
        } else if (result < 0) {
            printf("Error: Instruction execution failed\n");
            vm_state_destroy(state);
            return -1;
        }
        
        iterations++;
    }
    
    timer_stop(&timer);
    
    printf("\nExecution completed in %d iterations (%.3f ms)\n", iterations, timer_elapsed_ms(&timer));
    printf("\nFinal state:\n");
    vm_state_print(state);
    
    printf("\nMemory[0-15]:\n");
    for (int i = 0; i < 16; i++) {
        if (state->memory[i] != 0) {
            printf("  [%d] = %d\n", i, state->memory[i]);
        }
    }
    
    vm_state_destroy(state);
    return 0;
}

int main(int argc, char** argv) {
    printf("Bytecode VM Test Runner\n");
    printf("=======================\n");
    
    int run_add = 1;
    int run_branch = 1;
    int run_loop = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--add-only") == 0) {
            run_branch = 0;
            run_loop = 0;
        } else if (strcmp(argv[i], "--branch-only") == 0) {
            run_add = 0;
            run_loop = 0;
        } else if (strcmp(argv[i], "--loop-only") == 0) {
            run_add = 0;
            run_branch = 0;
        }
    }
    
    if (run_add) {
        BytecodeProgram* add_program = create_add_test();
        if (add_program) {
            test_program_host("ADD Test", add_program);
            bytecode_destroy(add_program);
        }
    }
    
    if (run_branch) {
        BytecodeProgram* branch_program = create_branch_test();
        if (branch_program) {
            test_program_host("Branch Test", branch_program);
            bytecode_destroy(branch_program);
        }
    }
    
    if (run_loop) {
        BytecodeProgram* loop_program = create_loop_test();
        if (loop_program) {
            test_program_host("Loop Test", loop_program);
            bytecode_destroy(loop_program);
        }
    }
    
    printf("\nAll tests completed.\n");
    return 0;
}

