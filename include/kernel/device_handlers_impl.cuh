#ifndef DEVICE_HANDLERS_IMPL_CUH
#define DEVICE_HANDLERS_IMPL_CUH

#include "device_instructions.cuh"
#include "device_handlers.cuh"

__device__ int32_t device_handler_nop(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)state; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_load_imm(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_dst)) {
        state->registers[reg_idx_dst] = EXTRACT_IMMEDIATE(op);
        device_update_flags(state, state->registers[reg_idx_dst]);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_add(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP(instr->operand, +);
    return 0;
}

__device__ int32_t device_handler_sub(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP(instr->operand, -);
    return 0;
}

__device__ int32_t device_handler_mul(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP(instr->operand, *);
    return 0;
}

__device__ int32_t device_handler_div(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP_SAFE(instr->operand, /);
    return 0;
}

__device__ int32_t device_handler_mod(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP_SAFE(instr->operand, %);
    return 0;
}

__device__ int32_t device_handler_and(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP(instr->operand, &);
    return 0;
}

__device__ int32_t device_handler_or(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP(instr->operand, |);
    return 0;
}

__device__ int32_t device_handler_xor(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_BINARY_OP(instr->operand, ^);
    return 0;
}

__device__ int32_t device_handler_not(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_idx_src1)) {
        int32_t result = ~state->registers[reg_idx_src1];
        state->registers[reg_idx_dst] = result;
        device_update_flags(state, result);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_shl(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op);
    int32_t reg_idx_src2 = EXTRACT_REG_SRC2(op);
    if (DEVICE_VALIDATE_REG_TRIPLE(reg_idx_dst, reg_idx_src1, reg_idx_src2)) {
        int32_t a = state->registers[reg_idx_src1];
        int32_t b = state->registers[reg_idx_src2];
        int32_t shift = b & 0x1F;
        int32_t result = a << shift;
        state->registers[reg_idx_dst] = result;
        device_update_flags(state, result);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_shr(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op);
    int32_t reg_idx_src2 = EXTRACT_REG_SRC2(op);
    if (DEVICE_VALIDATE_REG_TRIPLE(reg_idx_dst, reg_idx_src1, reg_idx_src2)) {
        int32_t a = state->registers[reg_idx_src1];
        int32_t b = state->registers[reg_idx_src2];
        int32_t shift = b & 0x1F;
        int32_t result = a >> shift;
        state->registers[reg_idx_dst] = result;
        device_update_flags(state, result);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_eq(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_COMPARE_OP(instr->operand, ==);
    return 0;
}

__device__ int32_t device_handler_ne(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_COMPARE_OP(instr->operand, !=);
    return 0;
}

__device__ int32_t device_handler_lt(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_COMPARE_OP(instr->operand, <);
    return 0;
}

__device__ int32_t device_handler_le(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_COMPARE_OP(instr->operand, <=);
    return 0;
}

__device__ int32_t device_handler_gt(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_COMPARE_OP(instr->operand, >);
    return 0;
}

__device__ int32_t device_handler_ge(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    DEVICE_COMPARE_OP(instr->operand, >=);
    return 0;
}

__device__ int32_t device_handler_load(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_dst)) {
        uint32_t addr = EXTRACT_ADDRESS(op);
        if (addr < memory_size) {
            state->registers[reg_idx_dst] = global_memory[addr];
            device_update_flags(state, state->registers[reg_idx_dst]);
            state->memory_access_count++;
        }
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_store(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1)) {
        uint32_t addr = EXTRACT_ADDRESS(op);
        if (addr < memory_size) {
            global_memory[addr] = state->registers[reg_idx_src1];
            state->memory_access_count++;
        }
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_load_shared(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_dst)) {
        uint32_t shared_addr = EXTRACT_ADDRESS(op);
        if (shared_addr < 1024) {
            state->registers[reg_idx_dst] = shared_memory[shared_addr];
            device_update_flags(state, state->registers[reg_idx_dst]);
            state->memory_access_count++;
        }
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_store_shared(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1)) {
        uint32_t shared_addr = EXTRACT_ADDRESS(op);
        if (shared_addr < 1024) {
            shared_memory[shared_addr] = state->registers[reg_idx_src1];
            state->memory_access_count++;
        }
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_jmp(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)state; (void)shared_memory; (void)global_memory; (void)memory_size;
    uint32_t op = instr->operand;
    uint32_t target = EXTRACT_JUMP_TARGET(op);
    if (target < program_size) {
        *pc = target;
    } else {
        (*pc)++;
    }
    return 0;
}

__device__ int32_t device_handler_jmp_true(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1)) {
        int32_t condition = state->registers[reg_idx_src1];
        uint32_t target = EXTRACT_JUMP_TARGET(op);
        if (target < program_size && condition != 0) {
            *pc = target;
            state->branch_count++;
        } else {
            (*pc)++;
        }
    } else {
        (*pc)++;
    }
    return 0;
}

__device__ int32_t device_handler_jmp_false(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1)) {
        int32_t condition = state->registers[reg_idx_src1];
        uint32_t target = EXTRACT_JUMP_TARGET(op);
        if (target < program_size && condition == 0) {
            *pc = target;
            state->branch_count++;
        } else {
            (*pc)++;
        }
    } else {
        (*pc)++;
    }
    return 0;
}

__device__ int32_t device_handler_push(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1) && state->stack_ptr < DEVICE_STACK_SIZE) {
        state->stack[state->stack_ptr++] = state->registers[reg_idx_src1];
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_pop(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    if (DEVICE_VALIDATE_REG(reg_idx_dst) && state->stack_ptr > 0) {
        state->registers[reg_idx_dst] = state->stack[--state->stack_ptr];
        device_update_flags(state, state->registers[reg_idx_dst]);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_dup(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    if (state->stack_ptr > 0 && state->stack_ptr < DEVICE_STACK_SIZE) {
        int32_t top = state->stack[state->stack_ptr - 1];
        state->stack[state->stack_ptr++] = top;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_swap(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    if (state->stack_ptr >= 2) {
        int32_t temp = state->stack[state->stack_ptr - 1];
        state->stack[state->stack_ptr - 1] = state->stack[state->stack_ptr - 2];
        state->stack[state->stack_ptr - 2] = temp;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_call(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size;
    uint32_t op = instr->operand;
    uint32_t target = EXTRACT_JUMP_TARGET(op);
    if (target < program_size && state->stack_ptr < DEVICE_STACK_SIZE) {
        state->stack[state->stack_ptr++] = (int32_t)(*pc + 1);
        *pc = target;
    } else {
        (*pc)++;
    }
    return 0;
}

__device__ int32_t device_handler_ret(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    if (state->stack_ptr > 0) {
        int32_t return_addr = state->stack[--state->stack_ptr];
        *pc = (uint32_t)return_addr;
    } else {
        return 1;
    }
    return 0;
}

__device__ int32_t device_handler_sync_warp(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)state; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    #if __CUDA_ARCH__ >= 700
    __syncwarp();
    #else
    __syncthreads();
    #endif
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_sync_block(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)state; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    __syncthreads();
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_ballot(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_cond = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_cond)) {
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
    return 0;
}

__device__ int32_t device_handler_shfl(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    uint32_t src_lane = EXTRACT_REG_SRC2(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
        int32_t value = state->registers[reg_src];
        #if __CUDA_ARCH__ >= 300
        value = __shfl_sync(0xFFFFFFFF, value, src_lane);
        #endif
        state->registers[reg_idx_dst] = value;
        device_update_flags(state, value);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_shfl_down(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    uint32_t delta = EXTRACT_REG_SRC2(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
        int32_t value = state->registers[reg_src];
        #if __CUDA_ARCH__ >= 300
        value = __shfl_down_sync(0xFFFFFFFF, value, delta);
        #endif
        state->registers[reg_idx_dst] = value;
        device_update_flags(state, value);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_shfl_up(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    uint32_t delta = EXTRACT_REG_SRC2(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
        int32_t value = state->registers[reg_src];
        #if __CUDA_ARCH__ >= 300
        value = __shfl_up_sync(0xFFFFFFFF, value, delta);
        #else
        /* Fallback: For thread 0, shuffle up returns own value (matches CUDA behavior) */
        #endif
        state->registers[reg_idx_dst] = value;
        device_update_flags(state, value);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_any_sync(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
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
    return 0;
}

__device__ int32_t device_handler_all_sync(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
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
    return 0;
}

__device__ int32_t device_handler_reduce_add(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
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
    return 0;
}

__device__ int32_t device_handler_reduce_max(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
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
    return 0;
}

__device__ int32_t device_handler_reduce_min(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_src = EXTRACT_REG_SRC1(op);
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_src)) {
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
    return 0;
}

__device__ int32_t device_handler_cmov(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_cond = EXTRACT_REG_SRC1(op);
    int32_t reg_src1 = (op >> 4) & 0xF;
    int32_t reg_src2 = op & 0xF;
    if (DEVICE_VALIDATE_REG(reg_idx_dst) && DEVICE_VALIDATE_REG(reg_cond) &&
        DEVICE_VALIDATE_REG(reg_src1) && DEVICE_VALIDATE_REG(reg_src2)) {
        int32_t condition = state->registers[reg_cond];
        int32_t result = (condition != 0) ? 
                         state->registers[reg_src1] : 
                         state->registers[reg_src2];
        state->registers[reg_idx_dst] = result;
        device_update_flags(state, result);
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_branch_hint(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)state; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_vload2(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    uint32_t base_addr = EXTRACT_ADDRESS(op);
    if (DEVICE_VALIDATE_REG(reg_idx_dst) && base_addr + 1 < memory_size) {
        int32_t* vec_ptr = &global_memory[base_addr];
        state->registers[reg_idx_dst] = vec_ptr[0];
        if (reg_idx_dst + 1 < DEVICE_MAX_REGISTERS) {
            state->registers[reg_idx_dst + 1] = vec_ptr[1];
        }
        device_update_flags(state, state->registers[reg_idx_dst]);
        state->memory_access_count += 2;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_vload4(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    uint32_t base_addr = EXTRACT_ADDRESS(op);
    if (DEVICE_VALIDATE_REG(reg_idx_dst) && base_addr + 3 < memory_size) {
        int32_t* vec_ptr = &global_memory[base_addr];
        for (int i = 0; i < 4 && (reg_idx_dst + i) < DEVICE_MAX_REGISTERS; i++) {
            state->registers[reg_idx_dst + i] = vec_ptr[i];
        }
        device_update_flags(state, state->registers[reg_idx_dst]);
        state->memory_access_count += 4;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_vstore2(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    uint32_t base_addr = EXTRACT_ADDRESS(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1) && base_addr + 1 < memory_size) {
        int32_t* vec_ptr = &global_memory[base_addr];
        vec_ptr[0] = state->registers[reg_idx_src1];
        if (reg_idx_src1 + 1 < DEVICE_MAX_REGISTERS) {
            vec_ptr[1] = state->registers[reg_idx_src1 + 1];
        }
        state->memory_access_count += 2;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_vstore4(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_src1 = EXTRACT_REG_DST(op);
    uint32_t base_addr = EXTRACT_ADDRESS(op);
    if (DEVICE_VALIDATE_REG(reg_idx_src1) && base_addr + 3 < memory_size) {
        int32_t* vec_ptr = &global_memory[base_addr];
        for (int i = 0; i < 4 && (reg_idx_src1 + i) < DEVICE_MAX_REGISTERS; i++) {
            vec_ptr[i] = state->registers[reg_idx_src1 + i];
        }
        state->memory_access_count += 4;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_prefetch(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    uint32_t base_addr = EXTRACT_ADDRESS(op);
    if (base_addr < memory_size) {
        #if __CUDA_ARCH__ >= 200
        volatile int32_t* prefetch_ptr = &global_memory[base_addr];
        (void)*prefetch_ptr;
        #endif
        state->memory_access_count++;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_load_string_ptr(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    uint32_t byte_offset = EXTRACT_ADDRESS(op);  
    if (DEVICE_VALIDATE_REG(reg_idx_dst)) {
        state->registers[reg_idx_dst] = (int32_t)byte_offset;
        device_update_flags(state, state->registers[reg_idx_dst]);
        state->memory_access_count++;
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_str_len(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_idx_src = EXTRACT_REG_SRC1(op);  
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_idx_src)) {
        uint32_t byte_offset = (uint32_t)state->registers[reg_idx_src];
        size_t max_bytes = memory_size * sizeof(int32_t);
        
        if (byte_offset < max_bytes) {
            const uint8_t* mem_bytes = (const uint8_t*)global_memory;
            const uint8_t* str_ptr = &mem_bytes[byte_offset];
            
            int32_t len = 0;
            while (byte_offset + len < max_bytes && str_ptr[len] != 0) {
                len++;
            }
            
            state->registers[reg_idx_dst] = len;
            device_update_flags(state, len);
            state->memory_access_count++;
        }
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_str_cmp(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);
    int32_t reg_idx_src1 = EXTRACT_REG_SRC1(op);  
    int32_t reg_idx_src2 = EXTRACT_REG_SRC2(op);  
    if (DEVICE_VALIDATE_REG_TRIPLE(reg_idx_dst, reg_idx_src1, reg_idx_src2)) {
        uint32_t offset1 = (uint32_t)state->registers[reg_idx_src1];
        uint32_t offset2 = (uint32_t)state->registers[reg_idx_src2];
        size_t max_bytes = memory_size * sizeof(int32_t);
        
        if (offset1 < max_bytes && offset2 < max_bytes) {
            const uint8_t* mem_bytes = (const uint8_t*)global_memory;
            const uint8_t* str1 = &mem_bytes[offset1];
            const uint8_t* str2 = &mem_bytes[offset2];
            int32_t result = 0;
            uint32_t i = 0;
            
            while (offset1 + i < max_bytes && offset2 + i < max_bytes) {
                uint8_t c1 = str1[i];
                uint8_t c2 = str2[i];
                
                if (c1 != c2) {
                    result = (c1 < c2) ? -1 : 1;
                    break;
                }
                
                if (c1 == 0) {
                    result = 0;  
                    break;
                }
                
                i++;
            }
            
            if (result == 0 && (offset1 + i >= max_bytes || offset2 + i >= max_bytes)) {
                if (offset1 + i >= max_bytes && offset2 + i >= max_bytes) {
                    result = 0;  
                } else if (offset1 + i >= max_bytes) {
                    if (offset2 + i < max_bytes && str2[i] == 0) {
                        result = 0;  
                    } else {
                        result = -1; 
                    }
                } else {
                    if (offset1 + i < max_bytes && str1[i] == 0) {
                        result = 0;  
                    } else {
                        result = 1;  
                    }
                }
            }
            
            state->registers[reg_idx_dst] = result;
            device_update_flags(state, result);
            state->memory_access_count += 2;
        }
    }
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_str_copy(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);  
    int32_t reg_idx_src = EXTRACT_REG_SRC1(op);  
    if (DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_idx_src)) {
        uint32_t dst_offset = (uint32_t)state->registers[reg_idx_dst];
        uint32_t src_offset = (uint32_t)state->registers[reg_idx_src];
        size_t max_bytes = memory_size * sizeof(int32_t);
        
        if (dst_offset < max_bytes && src_offset < max_bytes && dst_offset != src_offset) {
            uint8_t* mem_bytes = (uint8_t*)global_memory;
            const uint8_t* src = &mem_bytes[src_offset];
            uint8_t* dst = &mem_bytes[dst_offset];
            
            uint32_t i = 0;
            while (src_offset + i < max_bytes && dst_offset + i < max_bytes) {
                uint8_t byte = src[i];
                dst[i] = byte;
                
                if (byte == 0) {
                    break;
                }
                
                i++;
            }
            
            if (dst_offset + i < max_bytes && src_offset + i < max_bytes && src[i] != 0) {
                dst[i] = 0;
            }
            
            state->memory_access_count += 2;
        }
    }
    (*pc)++;
    return 0;
}

#define HEAP_ALIGNMENT 8
#define HEAP_HEADER_SIZE 2  
#define HEAP_FREE_LIST_HEAD 0  
#define HEAP_MIN_BLOCK_SIZE (HEAP_HEADER_SIZE + 1)
#define HEAP_START_OFFSET(mem_size) ((mem_size) * 3 / 4)
#define HEAP_SIZE(mem_size) ((mem_size) / 4)


__device__ inline uint32_t heap_get_block_size(int32_t* heap_base, uint32_t block_idx) {
    return (uint32_t)heap_base[block_idx];
}

__device__ inline void heap_set_block_size(int32_t* heap_base, uint32_t block_idx, uint32_t size) {
    heap_base[block_idx] = (int32_t)size;
}

__device__ inline int32_t heap_get_next_free(int32_t* heap_base, uint32_t block_idx) {
    return heap_base[block_idx + 1];
}

__device__ inline void heap_set_next_free(int32_t* heap_base, uint32_t block_idx, int32_t next) {
    heap_base[block_idx + 1] = next;
}

__device__ int32_t device_handler_malloc(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_dst = EXTRACT_REG_DST(op);  
    int32_t reg_idx_size = EXTRACT_REG_SRC1(op); 
    
    if (!DEVICE_VALIDATE_REG_PAIR(reg_idx_dst, reg_idx_size)) {
        (*pc)++;
        return 0;
    }
    
    uint32_t size_bytes = (uint32_t)state->registers[reg_idx_size];
    if (size_bytes == 0) {
        state->registers[reg_idx_dst] = 0;
        (*pc)++;
        return 0;
    }
    
    uint32_t heap_start = HEAP_START_OFFSET(memory_size);
    uint32_t heap_size = HEAP_SIZE(memory_size);
    int32_t* heap_base = &global_memory[heap_start];
    
    if (heap_size < HEAP_MIN_BLOCK_SIZE) {
        state->registers[reg_idx_dst] = 0;  
        (*pc)++;
        return 0;
    }
    
    uint32_t size_units = (size_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
    uint32_t block_size = size_units + HEAP_HEADER_SIZE; 
    
    if (block_size > heap_size - HEAP_HEADER_SIZE) {
        state->registers[reg_idx_dst] = 0;  
        (*pc)++;
        return 0;
    }
    
    uint32_t alloc_start = HEAP_HEADER_SIZE + 1;
    int32_t* alloc_ptr = &heap_base[HEAP_FREE_LIST_HEAD];
    atomicCAS((int*)alloc_ptr, 0, (int)alloc_start);
    uint32_t alloc_idx = (uint32_t)atomicAdd((int*)alloc_ptr, (int)block_size);
    
    if (alloc_idx == 0) {
        alloc_idx = alloc_start;
    } else if (alloc_idx < alloc_start) {
        alloc_idx = alloc_start;
    }
    
    if (alloc_idx + block_size > heap_size) {
        atomicSub((int*)alloc_ptr, (int)block_size);
        state->registers[reg_idx_dst] = 0;
        (*pc)++;
        return 0;
    }
    
    heap_set_block_size(heap_base, alloc_idx, block_size);
    heap_set_next_free(heap_base, alloc_idx, -1);
    uint32_t data_offset = heap_start + alloc_idx + HEAP_HEADER_SIZE;
    state->registers[reg_idx_dst] = (int32_t)(data_offset * sizeof(int32_t));  
    state->memory_access_count++;
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_free(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    uint32_t op = instr->operand;
    int32_t reg_idx_ptr = EXTRACT_REG_DST(op);
    
    if (!DEVICE_VALIDATE_REG(reg_idx_ptr)) {
        (*pc)++;
        return 0;
    }
    
    uint32_t ptr_bytes = (uint32_t)state->registers[reg_idx_ptr];
    if (ptr_bytes == 0) {
        (*pc)++;
        return 0;
    }
    
    uint32_t ptr_idx = ptr_bytes / sizeof(int32_t);
    uint32_t heap_start = HEAP_START_OFFSET(memory_size);
    uint32_t heap_size = HEAP_SIZE(memory_size);
    int32_t* heap_base = &global_memory[heap_start];
    
    if (ptr_idx < heap_start || ptr_idx >= heap_start + heap_size) {
        (*pc)++;
        return 0;
    }
    
    uint32_t data_idx = ptr_idx - heap_start;
    if (data_idx < HEAP_HEADER_SIZE) {
        (*pc)++;
        return 0;
    }
    
    uint32_t block_idx = data_idx - HEAP_HEADER_SIZE;
    
    if (block_idx >= heap_size) {
        (*pc)++;
        return 0;
    }
    
    uint32_t block_size = heap_get_block_size(heap_base, block_idx);
    heap_set_next_free(heap_base, block_idx, 0);  
    
    state->memory_access_count++;
    (*pc)++;
    return 0;
}

#define PRINT_BUFFER_START(mem_size) ((mem_size) * 2 / 3)  
#define PRINT_BUFFER_SIZE(mem_size) ((mem_size) / 12)  
#define PRINT_TYPE_INT 0
#define PRINT_TYPE_STR 1

__device__ int32_t device_handler_print(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)shared_memory; (void)program_size;
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id != 0) {
        (*pc)++;
        return 0;
    }
    
    uint32_t op = instr->operand;
    int32_t reg_idx_type = EXTRACT_REG_DST(op);  
    int32_t reg_idx_value = EXTRACT_REG_SRC1(op);  
    
    if (!DEVICE_VALIDATE_REG_PAIR(reg_idx_type, reg_idx_value)) {
        (*pc)++;
        return 0;
    }
    
    uint32_t print_type = (uint32_t)state->registers[reg_idx_type];
    uint32_t print_buffer_start = PRINT_BUFFER_START(memory_size);
    uint32_t print_buffer_size = PRINT_BUFFER_SIZE(memory_size);
    
    if (print_buffer_size < 3) {
        (*pc)++;
        return 0;
    }

    int32_t* slot_counter = &global_memory[print_buffer_start];
    atomicCAS((int*)slot_counter, 0, 3);
    uint32_t reserved_idx = (uint32_t)atomicAdd((int*)slot_counter, 3);
    
    if (reserved_idx == 0 || reserved_idx < 3) {
        reserved_idx = 3;
    }
    
    if (reserved_idx + 3 > print_buffer_size) {
        atomicSub((int*)slot_counter, 3);
        (*pc)++;
        return 0;
    }
    
    uint32_t slot_idx = reserved_idx;
    
    if (print_type == PRINT_TYPE_INT) {
        global_memory[print_buffer_start + slot_idx] = PRINT_TYPE_INT;
        global_memory[print_buffer_start + slot_idx + 1] = state->registers[reg_idx_value];
        global_memory[print_buffer_start + slot_idx + 2] = 0; 
    } else if (print_type == PRINT_TYPE_STR) {
        uint32_t str_offset_bytes = (uint32_t)state->registers[reg_idx_value];
        uint32_t str_offset = str_offset_bytes / sizeof(int32_t);
        
        if (str_offset < memory_size) {
            uint32_t len = 0;
            const uint8_t* str_bytes = (const uint8_t*)global_memory;
            size_t max_bytes = memory_size * sizeof(int32_t);
            
            while (str_offset_bytes + len < max_bytes && str_bytes[str_offset_bytes + len] != 0) {
                len++;
            }
            
            global_memory[print_buffer_start + slot_idx] = PRINT_TYPE_STR;
            global_memory[print_buffer_start + slot_idx + 1] = (int32_t)str_offset_bytes;
            global_memory[print_buffer_start + slot_idx + 2] = (int32_t)len;
        }
    }
    
    state->memory_access_count++;
    (*pc)++;
    return 0;
}

__device__ int32_t device_handler_halt(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)state; (void)shared_memory; (void)pc; (void)global_memory; (void)memory_size; (void)program_size;
    return 1;
}

__device__ int32_t device_handler_breakpoint(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
) {
    (void)instr; (void)state; (void)shared_memory; (void)global_memory; (void)memory_size; (void)program_size;
    (*pc)++;
    return 0;
}

#endif

