#include "vm/bytecode.h"
#include "utils/logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

BytecodeProgram* bytecode_create(size_t instruction_count, size_t data_size) {
    BytecodeProgram* program = (BytecodeProgram*)malloc(sizeof(BytecodeProgram));
    if (!program) return NULL;
    
    program->instruction_count = instruction_count;
    program->data_size = data_size;
    program->instructions = NULL;
    program->data_segment = NULL;
    
    if (instruction_count > 0) {
        program->instructions = (Instruction*)malloc(sizeof(Instruction) * instruction_count);
        if (!program->instructions) {
            free(program);
            return NULL;
        }
        memset(program->instructions, 0, sizeof(Instruction) * instruction_count);
    }
    
    if (data_size > 0) {
        program->data_segment = (uint8_t*)malloc(data_size);
        if (!program->data_segment) {
            free(program->instructions);
            free(program);
            return NULL;
        }
        memset(program->data_segment, 0, data_size);
    }
    
    return program;
}

void bytecode_destroy(BytecodeProgram* program) {
    if (!program) return;
    
    if (program->instructions) {
        free(program->instructions);
    }
    if (program->data_segment) {
        free(program->data_segment);
    }
    free(program);
}

int bytecode_load_from_file(const char* filename, BytecodeProgram* program) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return -1;
    }
    
    size_t instruction_count = 0;
    size_t data_size = 0;
    
    if (fread(&instruction_count, sizeof(size_t), 1, file) != 1 ||
        fread(&data_size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return -2;
    }
    
    BytecodeProgram* temp = bytecode_create(instruction_count, data_size);
    if (!temp) {
        fclose(file);
        return -3;
    }
    
    if (instruction_count > 0) {
        if (fread(temp->instructions, sizeof(Instruction), instruction_count, file) != instruction_count) {
            bytecode_destroy(temp);
            fclose(file);
            return -4;
        }
    }
    
    if (data_size > 0) {
        if (fread(temp->data_segment, 1, data_size, file) != data_size) {
            bytecode_destroy(temp);
            fclose(file);
            return -5;
        }
    }
    
    fclose(file);
    
    *program = *temp;
    free(temp);
    
    return 0;
}

int bytecode_save_to_file(const char* filename, const BytecodeProgram* program) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return -1;
    }
    
    if (fwrite(&program->instruction_count, sizeof(size_t), 1, file) != 1 ||
        fwrite(&program->data_size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return -2;
    }
    
    if (program->instruction_count > 0) {
        if (fwrite(program->instructions, sizeof(Instruction), program->instruction_count, file) != program->instruction_count) {
            fclose(file);
            return -3;
        }
    }
    
    if (program->data_size > 0 && program->data_segment) {
        if (fwrite(program->data_segment, 1, program->data_size, file) != program->data_size) {
            fclose(file);
            return -4;
        }
    }
    
    fclose(file);
    return 0;
}

void bytecode_print(const BytecodeProgram* program) {
    LOG_COLOR(COLOR_RESET, "Bytecode Program:\n");
    LOG_COLOR(COLOR_RESET, "  Instructions: %zu\n", program->instruction_count);
    LOG_COLOR(COLOR_RESET, "  Data size: %zu bytes\n", program->data_size);
}