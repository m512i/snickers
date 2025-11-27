#include "vm/instructions.h"
#include "vm/vm_state.h"
#include <stdio.h>
#include <string.h>

const char* instruction_get_name(OpCode opcode) {
    static const char* names[] = {
        "NOP", "ADD", "SUB", "MUL", "DIV", "MOD",
        "AND", "OR", "XOR", "NOT", "SHL", "SHR",
        "EQ", "NE", "LT", "LE", "GT", "GE",
        "LOAD_IMM", "LOAD", "STORE", "LOAD_SHARED", "STORE_SHARED",
        "PUSH", "POP", "DUP", "SWAP",
        "JMP", "JMP_TRUE", "JMP_FALSE", "CALL", "RET",
        "SYNC_WARP", "SYNC_BLOCK",
        "BALLOT", "SHFL", "SHFL_DOWN", "SHFL_UP", "ANY_SYNC", "ALL_SYNC",
        "REDUCE_ADD", "REDUCE_MAX", "REDUCE_MIN",
        "CMOV", "BRANCH_HINT",
        "VLOAD2", "VLOAD4", "VSTORE2", "VSTORE4",
        "PREFETCH",
        "LOAD_STRING_PTR", "STR_LEN", "STR_CMP", "STR_COPY",
        "MALLOC", "FREE", "PRINT",
        "HALT", "BREAKPOINT"
    };
    
    if (opcode < OP_MAX_OPCODE) {
        return names[opcode];
    }
    return "UNKNOWN";
}

int instruction_get_operand_count(OpCode opcode) {
    switch (opcode) {
        case OP_NOP:
        case OP_RET:
        case OP_HALT:
        case OP_POP:
        case OP_DUP:
        case OP_SWAP:
        case OP_NOT:
            return 0;
        
        case OP_LOAD_IMM:
        case OP_LOAD:
        case OP_STORE:
        case OP_JMP:
        case OP_JMP_TRUE:
        case OP_JMP_FALSE:
        case OP_CALL:
        case OP_PUSH:
        case OP_LOAD_STRING_PTR:
            return 1;
        
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_MOD:
        case OP_AND:
        case OP_OR:
        case OP_XOR:
        case OP_SHL:
        case OP_SHR:
        case OP_EQ:
        case OP_NE:
        case OP_LT:
        case OP_LE:
        case OP_GT:
        case OP_GE:
        case OP_STR_LEN:
        case OP_STR_CMP:
        case OP_STR_COPY:
        case OP_MALLOC:
        case OP_FREE:
        case OP_PRINT:
            return 1;  
        
        default:
            return 0;
    }
}

int instruction_execute_host(
    Instruction* instr,
    int32_t* registers,
    int32_t* stack,
    int* stack_ptr,
    int* pc,
    int32_t* memory,
    size_t memory_size
) {
    if (*stack_ptr < 0 || *stack_ptr >= VM_STACK_SIZE) {
        return -1;  
    }
    
    uint32_t op = instr->operand;
    int32_t a, b, result;
    
    switch (instr->opcode) {
        case OP_NOP:
            (*pc)++;
            break;
        
        case OP_LOAD_IMM:
            if ((op >> 16) < VM_MAX_REGISTERS) {
                registers[op >> 16] = (int32_t)(op & 0xFFFF);
            }
            (*pc)++;
            break;
        
        case OP_ADD:
            a = registers[(op >> 8) & 0xFF];
            b = registers[op & 0xFF];
            result = a + b;
            if ((op >> 16) < VM_MAX_REGISTERS) {
                registers[op >> 16] = result;
            }
            (*pc)++;
            break;
        
        case OP_SUB:
            a = registers[(op >> 8) & 0xFF];
            b = registers[op & 0xFF];
            result = a - b;
            if ((op >> 16) < VM_MAX_REGISTERS) {
                registers[op >> 16] = result;
            }
            (*pc)++;
            break;
        
        case OP_MUL:
            a = registers[(op >> 8) & 0xFF];
            b = registers[op & 0xFF];
            result = a * b;
            if ((op >> 16) < VM_MAX_REGISTERS) {
                registers[op >> 16] = result;
            }
            (*pc)++;
            break;
        
        case OP_DIV:
            a = registers[(op >> 8) & 0xFF];
            b = registers[op & 0xFF];
            if (b == 0) return -2;  
            result = a / b;
            if ((op >> 16) < VM_MAX_REGISTERS) {
                registers[op >> 16] = result;
            }
            (*pc)++;
            break;
        
        case OP_LOAD:
            if ((op >> 16) < VM_MAX_REGISTERS && (op & 0xFFFF) < memory_size) {
                registers[op >> 16] = memory[op & 0xFFFF];
            }
            (*pc)++;
            break;
        
        case OP_STORE:
            if ((op >> 16) < VM_MAX_REGISTERS && (op & 0xFFFF) < memory_size) {
                memory[op & 0xFFFF] = registers[op >> 16];
            }
            (*pc)++;
            break;
        
        case OP_JMP:
            *pc = (int)op;
            break;
        
        case OP_JMP_TRUE:
            if ((op >> 16) < VM_MAX_REGISTERS && registers[op >> 16] != 0) {
                *pc = (int)(op & 0xFFFF);
            } else {
                (*pc)++;
            }
            break;
        
        case OP_JMP_FALSE:
            if ((op >> 16) < VM_MAX_REGISTERS && registers[op >> 16] == 0) {
                *pc = (int)(op & 0xFFFF);
            } else {
                (*pc)++;
            }
            break;
        
        case OP_PUSH:
            if (*stack_ptr >= VM_STACK_SIZE) return -1;
            if ((op >> 16) < VM_MAX_REGISTERS) {
                stack[(*stack_ptr)++] = registers[op >> 16];
            }
            (*pc)++;
            break;
        
        case OP_POP:
            if (*stack_ptr <= 0) return -1;
            if ((op >> 16) < VM_MAX_REGISTERS) {
                registers[op >> 16] = stack[--(*stack_ptr)];
            }
            (*pc)++;
            break;
        
        case OP_MALLOC: {
            int32_t reg_dst = (op >> 16) & 0xFF;
            int32_t reg_size = (op >> 8) & 0xFF;
            
            if (reg_dst < VM_MAX_REGISTERS && reg_size < VM_MAX_REGISTERS) {
                uint32_t size_bytes = (uint32_t)registers[reg_size];
                size_t heap_start_idx = (memory_size * 3) / 4;
                size_t heap_size_units = memory_size / 4;
                
                uint32_t block_size = (size_bytes + sizeof(int32_t) - 1) / sizeof(int32_t) + 2; 
                uint32_t found = 0;
                
                if (heap_size_units >= (size_t)block_size + 1) {
                    uint32_t free_head = (uint32_t)memory[heap_start_idx];
                    uint32_t curr = (free_head > 0 && free_head < (uint32_t)heap_size_units) ? free_head : 1;
                    
                    while ((size_t)curr + (size_t)block_size < heap_size_units) {
                        int32_t mem_val = memory[heap_start_idx + curr];
                        uint32_t mem_val_u = (uint32_t)(mem_val >= 0 ? mem_val : 0);
                        if (mem_val == 0 || mem_val_u >= block_size) {
                            memory[heap_start_idx + curr] = (int32_t)block_size;
                            memory[heap_start_idx + curr + 1] = -1; 
                            registers[reg_dst] = (int32_t)((heap_start_idx + curr + 2) * sizeof(int32_t));
                            found = 1;
                            break;
                        }
                        int32_t next_ptr = memory[heap_start_idx + curr + 1];
                        if (next_ptr >= 0 && (uint32_t)next_ptr < (uint32_t)heap_size_units) {
                            curr = (uint32_t)next_ptr;
                        } else {
                            curr += block_size;
                        }
                    }
                }
                
                if (!found) {
                    registers[reg_dst] = 0;
                }
            }
            (*pc)++;
            break;
        }
        
        case OP_FREE: {
            int32_t reg_ptr = (op >> 16) & 0xFF;
            
            if (reg_ptr < VM_MAX_REGISTERS) {
                uint32_t ptr_bytes = (uint32_t)registers[reg_ptr];
                if (ptr_bytes != 0) {
                    size_t ptr_idx = ptr_bytes / sizeof(int32_t);
                    size_t heap_start_idx = (memory_size * 3) / 4;
                    size_t heap_size_units = memory_size / 4;
                    
                    if (ptr_idx > heap_start_idx + 2 && ptr_idx < heap_start_idx + heap_size_units) {
                        size_t block_idx = ptr_idx - heap_start_idx - 2;
                        if (block_idx < heap_size_units) {
                            int32_t old_head = memory[heap_start_idx];
                            memory[heap_start_idx] = (int32_t)block_idx;
                            memory[heap_start_idx + block_idx + 1] = old_head;
                        }
                    }
                }
            }
            (*pc)++;
            break;
        }
        
        case OP_PRINT: {
            int32_t reg_type = (op >> 16) & 0xFF;
            int32_t reg_value = (op >> 8) & 0xFF;
            
            if (reg_type < VM_MAX_REGISTERS && reg_value < VM_MAX_REGISTERS) {
                uint32_t print_type = (uint32_t)registers[reg_type];
                
                if (print_type == 0) {
                    printf("%d", registers[reg_value]);
                } else if (print_type == 1) {
                    uint32_t str_offset = (uint32_t)registers[reg_value];
                    uint32_t str_idx = str_offset / sizeof(int32_t);
                    
                    if (str_idx < memory_size) {
                        const uint8_t* str_bytes = (const uint8_t*)memory;
                        size_t max_bytes = memory_size * sizeof(int32_t);
                        
                        if (str_offset < max_bytes) {
                            const char* str = (const char*)&str_bytes[str_offset];
                            printf("%s", str);
                        }
                    }
                }
            }
            (*pc)++;
            break;
        }
        
        case OP_HALT:
            return 1;  
        
        default:
            return -3;  
    }
    
    return 0;
}