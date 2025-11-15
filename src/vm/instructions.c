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
        
        case OP_HALT:
            return 1;  
        
        default:
            return -3;  
    }
    
    return 0;
}