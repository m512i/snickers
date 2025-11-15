#ifndef INSTRUCTION_HANDLERS_H
#define INSTRUCTION_HANDLERS_H

#include <stdint.h>
#include <stddef.h>
#include "bytecode.h"

typedef struct DeviceVMState DeviceVMState;
typedef struct {
    int32_t* registers;
    int32_t* stack;
    int* stack_ptr;
    size_t max_registers;
    size_t stack_size;
} HostVMContext;

typedef int32_t (*DeviceInstructionHandler)(
    const Instruction* instr,
    DeviceVMState* state,
    int32_t* shared_memory,
    uint32_t* pc,
    int32_t* global_memory,
    size_t memory_size,
    size_t program_size
);

typedef int (*HostInstructionHandler)(
    const Instruction* instr,
    HostVMContext* ctx,
    int* pc,
    int32_t* memory,
    size_t memory_size
);

typedef struct {
    DeviceInstructionHandler device_handlers[OP_MAX_OPCODE];
    HostInstructionHandler host_handlers[OP_MAX_OPCODE];
    int initialized;
} InstructionHandlerRegistry;

void instruction_handlers_init(void);

DeviceInstructionHandler instruction_get_device_handler(OpCode opcode);

HostInstructionHandler instruction_get_host_handler(OpCode opcode);

#endif

