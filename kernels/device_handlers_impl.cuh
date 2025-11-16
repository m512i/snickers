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

