#include "vm/vm_state.h"
#include "utils/logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

VMState* vm_state_create(size_t memory_size) {
    VMState* state = (VMState*)malloc(sizeof(VMState));
    if (!state) return NULL;
    
    memset(state->registers, 0, sizeof(state->registers));
    memset(state->stack, 0, sizeof(state->stack));
    state->pc = 0;
    state->stack_ptr = 0;
    state->flags = 0;
    state->memory_size = memory_size;
    
    if (memory_size > 0) {
        state->memory = (int32_t*)malloc(sizeof(int32_t) * memory_size);
        if (!state->memory) {
            free(state);
            return NULL;
        }
        memset(state->memory, 0, sizeof(int32_t) * memory_size);
    } else {
        state->memory = NULL;
    }
    
    return state;
}

void vm_state_destroy(VMState* state) {
    if (!state) return;
    
    if (state->memory) {
        free(state->memory);
    }
    free(state);
}

void vm_state_reset(VMState* state) {
    if (!state) return;
    
    memset(state->registers, 0, sizeof(state->registers));
    memset(state->stack, 0, sizeof(state->stack));
    state->pc = 0;
    state->stack_ptr = 0;
    state->flags = 0;
    
    if (state->memory) {
        memset(state->memory, 0, sizeof(int32_t) * state->memory_size);
    }
}

void vm_state_set_flag(VMState* state, uint32_t flag, int value) {
    if (!state) return;
    
    if (value) {
        state->flags |= flag;
    } else {
        state->flags &= ~flag;
    }
}

int vm_state_get_flag(const VMState* state, uint32_t flag) {
    if (!state) return 0;
    return (state->flags & flag) != 0;
}

void vm_state_print(const VMState* state) {
    if (!state) return;
    
    LOG_COLOR(COLOR_RESET, "=== VM State ===\n");
    LOG_COLOR(COLOR_RESET, "PC: %u\n", state->pc);
    LOG_COLOR(COLOR_RESET, "Stack Ptr: %d\n", state->stack_ptr);
    LOG_COLOR(COLOR_RESET, "Flags: 0x%08X\n", state->flags);
    
    LOG_COLOR(COLOR_RESET, "Registers:\n");
    for (int i = 0; i < VM_MAX_REGISTERS; i++) {
        if (state->registers[i] != 0) {
            LOG_COLOR(COLOR_RESET, "  R%02d: 0x%08X (%d)\n", i, state->registers[i], state->registers[i]);
        }
    }
}

