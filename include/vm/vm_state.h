#ifndef VM_STATE_H
#define VM_STATE_H

#include <stdint.h>
#include <stddef.h>
#include "bytecode.h"

#define VM_MAX_REGISTERS 32
#define VM_STACK_SIZE 256

typedef struct {
    int32_t registers[VM_MAX_REGISTERS];
    uint32_t pc;
    int32_t stack[VM_STACK_SIZE];
    int32_t stack_ptr;
    uint32_t flags;
    int32_t* memory;
    size_t memory_size;
} VMState;

VMState* vm_state_create(size_t memory_size);
void vm_state_destroy(VMState* state);
void vm_state_reset(VMState* state);
void vm_state_print(const VMState* state);

#define FLAG_ZERO 0x01
#define FLAG_NEGATIVE 0x02
#define FLAG_CARRY 0x04
#define FLAG_OVERFLOW 0x08

void vm_state_set_flag(VMState* state, uint32_t flag, int value);
int vm_state_get_flag(const VMState* state, uint32_t flag);

#endif

