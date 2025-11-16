#ifndef DEVICE_HANDLERS_CUH
#define DEVICE_HANDLERS_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/vm/bytecode.h"
#include "../include/vm/operand_helpers.h"


#ifndef DEVICE_MAX_REGISTERS
#define DEVICE_MAX_REGISTERS 32
#endif
#ifndef DEVICE_STACK_SIZE
#define DEVICE_STACK_SIZE 256
#endif

typedef int32_t (*DeviceInstructionHandler)(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

#define DEVICE_VALIDATE_REG(reg_idx) IS_VALID_REG(reg_idx, DEVICE_MAX_REGISTERS)
#define DEVICE_VALIDATE_REG_PAIR(reg1, reg2) IS_VALID_REG_PAIR(reg1, reg2, DEVICE_MAX_REGISTERS)
#define DEVICE_VALIDATE_REG_TRIPLE(reg1, reg2, reg3) IS_VALID_REG_TRIPLE(reg1, reg2, reg3, DEVICE_MAX_REGISTERS)

#define DEVICE_BINARY_OP(op, operation) \
    do { \
        int32_t reg_idx_dst = EXTRACT_REG_DST(op); \
        int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op); \
        int32_t reg_idx_src2 = EXTRACT_REG_SRC2(op); \
        if (DEVICE_VALIDATE_REG_TRIPLE(reg_idx_dst, reg_idx_src1, reg_idx_src2)) { \
            int32_t a = state->registers[reg_idx_src1]; \
            int32_t b = state->registers[reg_idx_src2]; \
            int32_t result = (a) operation (b); \
            state->registers[reg_idx_dst] = result; \
            device_update_flags(state, result); \
        } \
        (*pc)++; \
    } while(0)

#define DEVICE_BINARY_OP_SAFE(op, operation) \
    do { \
        int32_t reg_idx_dst = EXTRACT_REG_DST(op); \
        int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op); \
        int32_t reg_idx_src2 = EXTRACT_REG_SRC2(op); \
        if (DEVICE_VALIDATE_REG_TRIPLE(reg_idx_dst, reg_idx_src1, reg_idx_src2)) { \
            int32_t a = state->registers[reg_idx_src1]; \
            int32_t b = state->registers[reg_idx_src2]; \
            if (b != 0) { \
                int32_t result = (a) operation (b); \
                state->registers[reg_idx_dst] = result; \
                device_update_flags(state, result); \
            } \
        } \
        (*pc)++; \
    } while(0)

#define DEVICE_COMPARE_OP(op, comparison) \
    do { \
        int32_t reg_idx_dst = EXTRACT_REG_DST(op); \
        int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op); \
        int32_t reg_idx_src2 = EXTRACT_REG_SRC2(op); \
        if (DEVICE_VALIDATE_REG_TRIPLE(reg_idx_dst, reg_idx_src1, reg_idx_src2)) { \
            int32_t a = state->registers[reg_idx_src1]; \
            int32_t b = state->registers[reg_idx_src2]; \
            int32_t result = ((a) comparison (b)) ? 1 : 0; \
            state->registers[reg_idx_dst] = result; \
            device_update_flags(state, result); \
        } \
        (*pc)++; \
    } while(0)

__device__ int32_t device_handler_nop(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_load_imm(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_add(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_sub(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_mul(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_div(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_mod(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_and(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_or(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_xor(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_not(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_shl(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_shr(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_eq(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_ne(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_lt(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_le(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_gt(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_ge(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_load(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_store(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_load_shared(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_store_shared(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_jmp(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_jmp_true(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_jmp_false(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_push(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_pop(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_dup(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_swap(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_call(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_ret(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_sync_warp(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_sync_block(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_ballot(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_shfl(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_shfl_down(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_shfl_up(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_any_sync(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_all_sync(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_reduce_add(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_reduce_max(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_reduce_min(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_cmov(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_branch_hint(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_vload2(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_vload4(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_vstore2(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_vstore4(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_prefetch(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_load_string_ptr(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_str_len(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_str_cmp(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_str_copy(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_halt(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

__device__ int32_t device_handler_breakpoint(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

#endif

