#ifndef INSTRUCTIONS_H
#define INSTRUCTIONS_H

#include <stdint.h>
#include "bytecode.h"

int instruction_execute_host(
    Instruction* instr,
    int32_t* registers,
    int32_t* stack,
    int* stack_ptr,
    int* pc,
    int32_t* memory,
    size_t memory_size
);

const char* instruction_get_name(OpCode opcode);

int instruction_get_operand_count(OpCode opcode);

#endif