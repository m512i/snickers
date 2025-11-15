#ifndef DEVICE_INSTRUCTIONS_CUH
#define DEVICE_INSTRUCTIONS_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/vm/bytecode.h"

#define DEVICE_MAX_REGISTERS 32
#define DEVICE_STACK_SIZE 256

typedef struct DeviceVMState {
    int32_t registers[DEVICE_MAX_REGISTERS];
    int32_t stack[DEVICE_STACK_SIZE];
    int32_t stack_ptr;
    uint32_t flags;
    
    uint32_t instruction_count;  
    uint32_t branch_count;       
    uint32_t memory_access_count; 
} DeviceVMState;

#include "device_handlers.cuh"

__device__ void device_update_flags(DeviceVMState* state, int32_t result);

__device__ inline void device_update_flags(DeviceVMState* state, int32_t result) {
    state->flags = 0;
    if (result == 0) {
        state->flags |= 0x01;  
    }
    if (result < 0) {
        state->flags |= 0x02;  
    }
}

__device__ int32_t device_execute_instruction(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size  
) {
    if (state->stack_ptr < 0) state->stack_ptr = 0;
    if (state->stack_ptr >= DEVICE_STACK_SIZE) state->stack_ptr = DEVICE_STACK_SIZE - 1;
    
    DeviceInstructionHandler handler = NULL;
    
    switch (instr->opcode) {
        case OP_NOP: handler = device_handler_nop; break;
        case OP_LOAD_IMM: handler = device_handler_load_imm; break;
        case OP_ADD: handler = device_handler_add; break;
        case OP_SUB: handler = device_handler_sub; break;
        case OP_MUL: handler = device_handler_mul; break;
        case OP_DIV: handler = device_handler_div; break;
        case OP_MOD: handler = device_handler_mod; break;
        case OP_AND: handler = device_handler_and; break;
        case OP_OR: handler = device_handler_or; break;
        case OP_XOR: handler = device_handler_xor; break;
        case OP_NOT: handler = device_handler_not; break;
        case OP_SHL: handler = device_handler_shl; break;
        case OP_SHR: handler = device_handler_shr; break;
        case OP_EQ: handler = device_handler_eq; break;
        case OP_NE: handler = device_handler_ne; break;
        case OP_LT: handler = device_handler_lt; break;
        case OP_LE: handler = device_handler_le; break;
        case OP_GT: handler = device_handler_gt; break;
        case OP_GE: handler = device_handler_ge; break;
        case OP_LOAD: handler = device_handler_load; break;
        case OP_STORE: handler = device_handler_store; break;
        case OP_LOAD_SHARED: handler = device_handler_load_shared; break;
        case OP_STORE_SHARED: handler = device_handler_store_shared; break;
        case OP_JMP: handler = device_handler_jmp; break;
        case OP_JMP_TRUE: handler = device_handler_jmp_true; break;
        case OP_JMP_FALSE: handler = device_handler_jmp_false; break;
        case OP_PUSH: handler = device_handler_push; break;
        case OP_POP: handler = device_handler_pop; break;
        case OP_DUP: handler = device_handler_dup; break;
        case OP_SWAP: handler = device_handler_swap; break;
        case OP_CALL: handler = device_handler_call; break;
        case OP_RET: handler = device_handler_ret; break;
        case OP_SYNC_WARP: handler = device_handler_sync_warp; break;
        case OP_SYNC_BLOCK: handler = device_handler_sync_block; break;
        case OP_BALLOT: handler = device_handler_ballot; break;
        case OP_SHFL: handler = device_handler_shfl; break;
        case OP_SHFL_DOWN: handler = device_handler_shfl_down; break;
        case OP_SHFL_UP: handler = device_handler_shfl_up; break;
        case OP_ANY_SYNC: handler = device_handler_any_sync; break;
        case OP_ALL_SYNC: handler = device_handler_all_sync; break;
        case OP_REDUCE_ADD: handler = device_handler_reduce_add; break;
        case OP_REDUCE_MAX: handler = device_handler_reduce_max; break;
        case OP_REDUCE_MIN: handler = device_handler_reduce_min; break;
        case OP_CMOV: handler = device_handler_cmov; break;
        case OP_BRANCH_HINT: handler = device_handler_branch_hint; break;
        case OP_VLOAD2: handler = device_handler_vload2; break;
        case OP_VLOAD4: handler = device_handler_vload4; break;
        case OP_VSTORE2: handler = device_handler_vstore2; break;
        case OP_VSTORE4: handler = device_handler_vstore4; break;
        case OP_PREFETCH: handler = device_handler_prefetch; break;
        case OP_HALT: handler = device_handler_halt; break;
        case OP_BREAKPOINT: handler = device_handler_breakpoint; break;
        default:
            (*pc)++;
            state->instruction_count++;
            return 0;
    }
    
    if (handler != NULL) {
        int32_t result = handler(instr, state, shared_memory, pc, global_memory, memory_size, program_size);
        state->instruction_count++;
        return result;
    }
    
    (*pc)++;
    state->instruction_count++;
    return 0;
}

__device__ int32_t device_execute_instruction(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size
) {
    return device_execute_instruction(instr, state, shared_memory, pc, global_memory, memory_size, 0);
}

#endif
