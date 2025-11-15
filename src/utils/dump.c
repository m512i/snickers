#include "utils/dump.h"
#include "utils/colors.h"
#include "vm/instructions.h"
#include <stdio.h>
#include <stdint.h>

void dump_instruction(const Instruction* instr, int index) {
    printf(COLOR_METADATA "%04d: " COLOR_INFO "%-12s" COLOR_RESET, index, instruction_get_name(instr->opcode));
    if (instruction_get_operand_count(instr->opcode) > 0) {
        printf(" " COLOR_VALUE "0x%08X" COLOR_METADATA " (%d)" COLOR_RESET, instr->operand, instr->operand);
    }
    printf("\n");
}

void dump_program(const BytecodeProgram* program) {
    printf(COLOR_HEADER "=== Bytecode Program ===" COLOR_RESET "\n");
    printf(COLOR_LABEL "Instructions:" COLOR_RESET " " COLOR_VALUE "%zu" COLOR_RESET "\n", program->instruction_count);
    printf(COLOR_LABEL "Data size:" COLOR_RESET " " COLOR_VALUE "%zu bytes" COLOR_RESET "\n", program->data_size);
    printf("\n" COLOR_LABEL "Instructions:" COLOR_RESET "\n");
    
    for (size_t i = 0; i < program->instruction_count; i++) {
        dump_instruction(&program->instructions[i], (int)i);
    }
    
    if (program->data_size > 0 && program->data_segment != NULL) {
        printf("\nData segment:\n");
        hex_dump(program->data_segment, program->data_size);
    }
    printf("\n");
}

void dump_memory(const int32_t* memory, size_t size, size_t start_addr) {
    printf("=== Memory Dump (starting at 0x%zX) ===\n", start_addr);
    for (size_t i = 0; i < size; i += 8) {
        printf("0x%04zX: ", start_addr + i);
        for (size_t j = 0; j < 8 && (i + j) < size; j++) {
            printf("%08X ", memory[i + j]);
        }
        printf("\n");
    }
}

void dump_registers(const int32_t* registers, size_t count) {
    printf("=== Registers ===\n");
    for (size_t i = 0; i < count; i++) {
        printf("R%02zu: 0x%08X (%d)\n", i, registers[i], registers[i]);
    }
}

void dump_stack(const int32_t* stack, int stack_ptr) {
    printf("=== Stack (ptr: %d) ===\n", stack_ptr);
    if (stack_ptr <= 0) {
        printf("(empty)\n");
        return;
    }
    
    for (int i = stack_ptr - 1; i >= 0; i--) {
        printf("  [%d] 0x%08X (%d)\n", i, stack[i], stack[i]);
    }
}

void hex_dump(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i += 16) {
        printf("%04zX: ", i);
        
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            printf("%02X ", bytes[i + j]);
        }
        
        for (size_t j = size - i; j < 16; j++) {
            printf("   ");
        }
        
        printf(" | ");
        
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            char c = bytes[i + j];
            printf("%c", (c >= 32 && c < 127) ? c : '.');
        }
        
        printf("\n");
    }
}