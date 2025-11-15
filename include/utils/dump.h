#ifndef DUMP_H
#define DUMP_H

#include <stdint.h>
#include <stddef.h>
#include "../vm/bytecode.h"

void dump_instruction(const Instruction* instr, int index);
void dump_program(const BytecodeProgram* program);
void dump_memory(const int32_t* memory, size_t size, size_t start_addr);
void dump_registers(const int32_t* registers, size_t count);
void dump_stack(const int32_t* stack, int stack_ptr);

#ifdef __cplusplus
extern "C" {
#endif
    void hex_dump(const void* data, size_t size);
#ifdef __cplusplus
}
#endif

#endif