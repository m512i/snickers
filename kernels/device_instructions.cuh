#ifndef DEVICE_INSTRUCTIONS_CUH
#define DEVICE_INSTRUCTIONS_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/vm/bytecode.h"

#define DEVICE_MAX_REGISTERS 32
#define DEVICE_STACK_SIZE 256

typedef struct {
    int32_t registers[DEVICE_MAX_REGISTERS];
    int32_t stack[DEVICE_STACK_SIZE];
    int32_t stack_ptr;
    uint32_t flags;
    
    uint32_t instruction_count;  
    uint32_t branch_count;       
    uint32_t memory_access_count; 
} DeviceVMState;

__device__ int32_t device_execute_instruction(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size
);

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
    uint32_t op = instr->operand;
    int32_t a, b, result = 0;
    int32_t reg_idx_dst, reg_idx_src1, reg_idx_src2;
    
    if (state->stack_ptr < 0) state->stack_ptr = 0;
    if (state->stack_ptr >= DEVICE_STACK_SIZE) state->stack_ptr = DEVICE_STACK_SIZE - 1;
    
    switch (instr->opcode) {
        case OP_NOP:
            (*pc)++;
            break;
        
        case OP_LOAD_IMM:
            reg_idx_dst = (op >> 16) & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS) {
                state->registers[reg_idx_dst] = (int32_t)(op & 0xFFFF);
                device_update_flags(state, state->registers[reg_idx_dst]);
            }
            (*pc)++;
            break;
        
        case OP_ADD:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = a + b;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_SUB:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = a - b;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_MUL:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = a * b;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_DIV:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                if (b != 0) {
                    result = a / b;
                    state->registers[reg_idx_dst] = result;
                    device_update_flags(state, result);
                }
            }
            (*pc)++;
            break;
        
        case OP_MOD:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                if (b != 0) {
                    result = a % b;
                    state->registers[reg_idx_dst] = result;
                    device_update_flags(state, result);
                }
            }
            (*pc)++;
            break;
        
        case OP_AND:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = a & b;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_OR:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = a | b;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_XOR:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = a ^ b;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_NOT:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                result = ~a;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_SHL:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                int32_t shift = b & 0x1F;
                result = a << shift;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_SHR:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                int32_t shift = b & 0x1F;
                result = a >> shift;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_EQ:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = (a == b) ? 1 : 0;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_NE:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = (a != b) ? 1 : 0;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_LT:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = (a < b) ? 1 : 0;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_LE:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = (a <= b) ? 1 : 0;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_GT:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = (a > b) ? 1 : 0;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_GE:
            reg_idx_dst = (op >> 16) & 0xFF;
            reg_idx_src1 = (op >> 8) & 0xFF;
            reg_idx_src2 = op & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS &&
                reg_idx_src1 < DEVICE_MAX_REGISTERS &&
                reg_idx_src2 < DEVICE_MAX_REGISTERS) {
                a = state->registers[reg_idx_src1];
                b = state->registers[reg_idx_src2];
                result = (a >= b) ? 1 : 0;
                state->registers[reg_idx_dst] = result;
                device_update_flags(state, result);
            }
            (*pc)++;
            break;
        
        case OP_LOAD:
            reg_idx_dst = (op >> 16) & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS) {
                uint32_t addr = op & 0xFFFF;
                if (addr < memory_size) {
                    state->registers[reg_idx_dst] = global_memory[addr];
                    device_update_flags(state, state->registers[reg_idx_dst]);
                    state->memory_access_count++;
                }
            }
            (*pc)++;
            break;
        
        case OP_STORE:
            reg_idx_src1 = (op >> 16) & 0xFF;
            if (reg_idx_src1 < DEVICE_MAX_REGISTERS) {
                uint32_t addr = op & 0xFFFF;
                if (addr < memory_size) {
                    global_memory[addr] = state->registers[reg_idx_src1];
                    state->memory_access_count++;
                }
            }
            (*pc)++;
            break;
        
        case OP_LOAD_SHARED:
            reg_idx_dst = (op >> 16) & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS) {
                uint32_t shared_addr = op & 0xFFFF;
                if (shared_addr < 1024) {  
                    state->registers[reg_idx_dst] = shared_memory[shared_addr];
                    device_update_flags(state, state->registers[reg_idx_dst]);
                    state->memory_access_count++;
                }
            }
            (*pc)++;
            break;
        
        case OP_STORE_SHARED:
            reg_idx_src1 = (op >> 16) & 0xFF;
            if (reg_idx_src1 < DEVICE_MAX_REGISTERS) {
                uint32_t shared_addr = op & 0xFFFF;
                if (shared_addr < 1024) {  
                    shared_memory[shared_addr] = state->registers[reg_idx_src1];
                    state->memory_access_count++;
                }
            }
            (*pc)++;
            break;
        
        case OP_JMP:
            {
                uint32_t target = op & 0xFFFF;
                if (target < program_size) {
                    *pc = target;
                } else {
                    (*pc)++;
                }
            }
            break;
        
        case OP_JMP_TRUE:
            reg_idx_src1 = (op >> 16) & 0xFF;
            if (reg_idx_src1 < DEVICE_MAX_REGISTERS) {
                int32_t condition = state->registers[reg_idx_src1];
                uint32_t target = op & 0xFFFF;
                if (target < program_size && condition != 0) {
                    *pc = target;
                    state->branch_count++;
                } else {
                    (*pc)++;
                }
            } else {
                (*pc)++;
            }
            break;
        
        case OP_JMP_FALSE:
            reg_idx_src1 = (op >> 16) & 0xFF;
            if (reg_idx_src1 < DEVICE_MAX_REGISTERS) {
                int32_t condition = state->registers[reg_idx_src1];
                uint32_t target = op & 0xFFFF;
                if (target < program_size && condition == 0) {
                    *pc = target;
                    state->branch_count++;
                } else {
                    (*pc)++;
                }
            } else {
                (*pc)++;
            }
            break;
        
        case OP_PUSH:
            reg_idx_src1 = (op >> 16) & 0xFF;
            if (reg_idx_src1 < DEVICE_MAX_REGISTERS && state->stack_ptr < DEVICE_STACK_SIZE) {
                state->stack[state->stack_ptr++] = state->registers[reg_idx_src1];
            }
            (*pc)++;
            break;
        
        case OP_POP:
            reg_idx_dst = (op >> 16) & 0xFF;
            if (reg_idx_dst < DEVICE_MAX_REGISTERS && state->stack_ptr > 0) {
                state->registers[reg_idx_dst] = state->stack[--state->stack_ptr];
                device_update_flags(state, state->registers[reg_idx_dst]);
            }
            (*pc)++;
            break;
        
        case OP_DUP:
            if (state->stack_ptr > 0 && state->stack_ptr < DEVICE_STACK_SIZE) {
                int32_t top = state->stack[state->stack_ptr - 1];
                state->stack[state->stack_ptr++] = top;
            }
            (*pc)++;
            break;
        
        case OP_SWAP:
            if (state->stack_ptr >= 2) {
                int32_t temp = state->stack[state->stack_ptr - 1];
                state->stack[state->stack_ptr - 1] = state->stack[state->stack_ptr - 2];
                state->stack[state->stack_ptr - 2] = temp;
            }
            (*pc)++;
            break;
        
        case OP_CALL:
            {
                uint32_t target = op & 0xFFFF;
                if (target < program_size && state->stack_ptr < DEVICE_STACK_SIZE) {
                    state->stack[state->stack_ptr++] = (int32_t)(*pc + 1);
                    *pc = target;
                } else {
                    (*pc)++;
                }
            }
            break;
        
        case OP_RET:
            if (state->stack_ptr > 0) {
                int32_t return_addr = state->stack[--state->stack_ptr];
                *pc = (uint32_t)return_addr;
            } else {
                return 1;
            }
            break;
        
        case OP_SYNC_WARP:
            #if __CUDA_ARCH__ >= 700
            __syncwarp();  
            #else
            __syncthreads();  
            #endif
            (*pc)++;
            break;
        
        case OP_SYNC_BLOCK:
            __syncthreads();
            (*pc)++;
            break;
        
        case OP_BREAKPOINT:
            /* Debug breakpoint - in debug builds, this can trigger instrumentation
             For now, it's a no-op but increments instruction count
             Can be extended to check debug flags or trigger host-side callbacks */
            (*pc)++;
            break;
        
        case OP_BALLOT:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_cond = (op >> 8) & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_cond < DEVICE_MAX_REGISTERS) {
                    int32_t condition = state->registers[reg_cond];
                    #if __CUDA_ARCH__ >= 700
                    unsigned mask = __ballot_sync(0xFFFFFFFF, condition != 0);
                    state->registers[reg_idx_dst] = (int32_t)mask;
                    #else
                    state->registers[reg_idx_dst] = (condition != 0) ? 1 : 0;
                    #endif
                    device_update_flags(state, state->registers[reg_idx_dst]);
                }
                (*pc)++;
            }
            break;
        
        case OP_SHFL:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                uint32_t src_lane = op & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 300
                    value = __shfl_sync(0xFFFFFFFF, value, src_lane);
                    #endif
                    state->registers[reg_idx_dst] = value;
                    device_update_flags(state, value);
                }
                (*pc)++;
            }
            break;
        
        case OP_SHFL_DOWN:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                uint32_t delta = op & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 300
                    value = __shfl_down_sync(0xFFFFFFFF, value, delta);
                    #endif
                    state->registers[reg_idx_dst] = value;
                    device_update_flags(state, value);
                }
                (*pc)++;
            }
            break;
        
        case OP_SHFL_UP:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                uint32_t delta = op & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 300
                    value = __shfl_up_sync(0xFFFFFFFF, value, delta);
                    #else
                    /* Fallback: For thread 0, shuffle up returns own value (matches CUDA behavior)
                     For other threads, we'd need shared memory to simulate, but for simplicity
                     we just use the source value (which is correct for thread 0)
                     This ensures the instruction always executes and updates the destination */
                    #endif
                    state->registers[reg_idx_dst] = value;
                    device_update_flags(state, value);
                }
                (*pc)++;
            }
            break;
        
        case OP_ANY_SYNC:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 700
                    int result = __any_sync(0xFFFFFFFF, value != 0) ? 1 : 0;
                    #else
                    int result = (value != 0) ? 1 : 0;
                    #endif
                    state->registers[reg_idx_dst] = result;
                    device_update_flags(state, result);
                }
                (*pc)++;
            }
            break;
        
        case OP_ALL_SYNC:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 700
                    int result = __all_sync(0xFFFFFFFF, value != 0) ? 1 : 0;
                    #else
                    int result = (value != 0) ? 1 : 0;
                    #endif
                    state->registers[reg_idx_dst] = result;
                    device_update_flags(state, result);
                }
                (*pc)++;
            }
            break;
        
        case OP_REDUCE_ADD:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 300
                    #pragma unroll
                    for (int i = 16; i > 0; i /= 2) {
                        int32_t other = __shfl_down_sync(0xFFFFFFFF, value, i);
                        value += other;
                    }
                    #endif
                    state->registers[reg_idx_dst] = value;
                    device_update_flags(state, value);
                }
                (*pc)++;
            }
            break;
        
        case OP_REDUCE_MAX:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 300
                    #pragma unroll
                    for (int i = 16; i > 0; i /= 2) {
                        int32_t other = __shfl_down_sync(0xFFFFFFFF, value, i);
                        value = (value > other) ? value : other;
                    }
                    #endif
                    state->registers[reg_idx_dst] = value;
                    device_update_flags(state, value);
                }
                (*pc)++;
            }
            break;
        
        case OP_REDUCE_MIN:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_src = (op >> 8) & 0xFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && reg_src < DEVICE_MAX_REGISTERS) {
                    int32_t value = state->registers[reg_src];
                    #if __CUDA_ARCH__ >= 300
                    #pragma unroll
                    for (int i = 16; i > 0; i /= 2) {
                        int32_t other = __shfl_down_sync(0xFFFFFFFF, value, i);
                        value = (value < other) ? value : other;
                    }
                    #endif
                    state->registers[reg_idx_dst] = value;
                    device_update_flags(state, value);
                }
                (*pc)++;
            }
            break;
        
        case OP_CMOV:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t reg_cond = (op >> 8) & 0xFF;
                uint32_t reg_src1 = (op >> 4) & 0xF;
                uint32_t reg_src2 = op & 0xF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && 
                    reg_cond < DEVICE_MAX_REGISTERS &&
                    reg_src1 < DEVICE_MAX_REGISTERS && 
                    reg_src2 < DEVICE_MAX_REGISTERS) {
                    int32_t condition = state->registers[reg_cond];
                    int32_t result = (condition != 0) ? 
                                     state->registers[reg_src1] : 
                                     state->registers[reg_src2];
                    state->registers[reg_idx_dst] = result;
                    device_update_flags(state, result);
                }
                (*pc)++;
            }
            break;
        
        case OP_BRANCH_HINT:
            // Branch hint - no-op instruction used for optimization hints
            // Can be used to mark likely/unlikely branches
            (*pc)++;
            break;
        
        case OP_VLOAD2:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t base_addr = op & 0xFFFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && base_addr + 1 < memory_size) {
                    int32_t* vec_ptr = &global_memory[base_addr];
                    state->registers[reg_idx_dst] = vec_ptr[0];
                    if (reg_idx_dst + 1 < DEVICE_MAX_REGISTERS) {
                        state->registers[reg_idx_dst + 1] = vec_ptr[1];
                    }
                    device_update_flags(state, state->registers[reg_idx_dst]);
                    state->memory_access_count += 2;
                }
                (*pc)++;
            }
            break;
        
        case OP_VLOAD4:
            {
                reg_idx_dst = (op >> 16) & 0xFF;
                uint32_t base_addr = op & 0xFFFF;
                if (reg_idx_dst < DEVICE_MAX_REGISTERS && base_addr + 3 < memory_size) {
                    int32_t* vec_ptr = &global_memory[base_addr];
                    for (int i = 0; i < 4 && (reg_idx_dst + i) < DEVICE_MAX_REGISTERS; i++) {
                        state->registers[reg_idx_dst + i] = vec_ptr[i];
                    }
                    device_update_flags(state, state->registers[reg_idx_dst]);
                    state->memory_access_count += 4;
                }
                (*pc)++;
            }
            break;
        
        case OP_VSTORE2:
            {
                reg_idx_src1 = (op >> 16) & 0xFF;
                uint32_t base_addr = op & 0xFFFF;
                if (reg_idx_src1 < DEVICE_MAX_REGISTERS && base_addr + 1 < memory_size) {
                    int32_t* vec_ptr = &global_memory[base_addr];
                    vec_ptr[0] = state->registers[reg_idx_src1];
                    if (reg_idx_src1 + 1 < DEVICE_MAX_REGISTERS) {
                        vec_ptr[1] = state->registers[reg_idx_src1 + 1];
                    }
                    state->memory_access_count += 2;
                }
                (*pc)++;
            }
            break;
        
        case OP_VSTORE4:
            {
                reg_idx_src1 = (op >> 16) & 0xFF;
                uint32_t base_addr = op & 0xFFFF;
                if (reg_idx_src1 < DEVICE_MAX_REGISTERS && base_addr + 3 < memory_size) {
                    int32_t* vec_ptr = &global_memory[base_addr];
                    for (int i = 0; i < 4 && (reg_idx_src1 + i) < DEVICE_MAX_REGISTERS; i++) {
                        vec_ptr[i] = state->registers[reg_idx_src1 + i];
                    }
                    state->memory_access_count += 4;
                }
                (*pc)++;
            }
            break;
        
        case OP_PREFETCH:
            {
                uint32_t base_addr = op & 0xFFFF;
                if (base_addr < memory_size) {
                    #if __CUDA_ARCH__ >= 200
                    volatile int32_t* prefetch_ptr = &global_memory[base_addr];
                    (void)*prefetch_ptr;  
                    #endif
                    state->memory_access_count++;
                }
                (*pc)++;
            }
            break;
        
        case OP_HALT:
            return 1;  
        
        default:
            (*pc)++;  
            break;
    }
    
    state->instruction_count++;
    
    return 0;
}

#endif