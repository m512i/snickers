#include "utils/dump.h"
#include "utils/logger.h"
#include "vm/instructions.h"
#include <stdio.h>
#include <stdint.h>

void dump_instruction(const Instruction* instr, int index) {
    LOG_COLOR(COLOR_METADATA, "%04d: ", index);
    LOG_COLOR(COLOR_INFO, "%-12s", instruction_get_name(instr->opcode));
    if (instruction_get_operand_count(instr->opcode) > 0) {
        LOG_COLOR(COLOR_RESET, " ");
        LOG_COLOR(COLOR_VALUE, "0x%08X", instr->operand);
        LOG_COLOR(COLOR_METADATA, " (%d)", instr->operand);
    }
    LOG_COLOR(COLOR_RESET, "\n");
}

void dump_program(const BytecodeProgram* program) {
    LOG_COLOR(COLOR_HEADER, "=== Bytecode Program ===\n");
    LOG_COLOR(COLOR_LABEL, "Instructions:");
    LOG_COLOR(COLOR_RESET, " ");
    LOG_COLOR(COLOR_VALUE, "%zu\n", program->instruction_count);
    LOG_COLOR(COLOR_LABEL, "Data size:");
    LOG_COLOR(COLOR_RESET, " ");
    LOG_COLOR(COLOR_VALUE, "%zu bytes\n", program->data_size);
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_COLOR(COLOR_LABEL, "Instructions:\n");
    
    for (size_t i = 0; i < program->instruction_count; i++) {
        dump_instruction(&program->instructions[i], (int)i);
    }
    
    if (program->data_size > 0 && program->data_segment != NULL) {
        LOG_COLOR(COLOR_RESET, "\nData segment:\n");
        hex_dump(program->data_segment, program->data_size);
    }
    LOG_COLOR(COLOR_RESET, "\n");
}

void dump_memory(const int32_t* memory, size_t size, size_t start_addr) {
    LOG_COLOR(COLOR_RESET, "=== Memory Dump (starting at 0x%zX) ===\n", start_addr);
    for (size_t i = 0; i < size; i += 8) {
        LOG_COLOR(COLOR_RESET, "0x%04zX: ", start_addr + i);
        for (size_t j = 0; j < 8 && (i + j) < size; j++) {
            LOG_COLOR(COLOR_RESET, "%08X ", memory[i + j]);
        }
        LOG_COLOR(COLOR_RESET, "\n");
    }
}

void dump_registers(const int32_t* registers, size_t count) {
    LOG_COLOR(COLOR_RESET, "=== Registers ===\n");
    for (size_t i = 0; i < count; i++) {
        LOG_COLOR(COLOR_RESET, "R%02zu: 0x%08X (%d)\n", i, registers[i], registers[i]);
    }
}

void dump_stack(const int32_t* stack, int stack_ptr) {
    LOG_COLOR(COLOR_RESET, "=== Stack (ptr: %d) ===\n", stack_ptr);
    if (stack_ptr <= 0) {
        LOG_COLOR(COLOR_RESET, "(empty)\n");
        return;
    }
    
    for (int i = stack_ptr - 1; i >= 0; i--) {
        LOG_COLOR(COLOR_RESET, "  [%d] 0x%08X (%d)\n", i, stack[i], stack[i]);
    }
}

void hex_dump(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i += 16) {
        LOG_COLOR(COLOR_RESET, "%04zX: ", i);
        
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            LOG_COLOR(COLOR_RESET, "%02X ", bytes[i + j]);
        }
        
        for (size_t j = size - i; j < 16; j++) {
            LOG_COLOR(COLOR_RESET, "   ");
        }
        
        LOG_COLOR(COLOR_RESET, " | ");
        
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            char c = bytes[i + j];
            LOG_COLOR(COLOR_RESET, "%c", (c >= 32 && c < 127) ? c : '.');
        }
        
        LOG_COLOR(COLOR_RESET, "\n");
    }
}