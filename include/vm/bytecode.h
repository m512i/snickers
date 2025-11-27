#ifndef BYTECODE_H
#define BYTECODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OP_NOP = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    OP_AND,
    OP_OR,
    OP_XOR,
    OP_NOT,
    OP_SHL,  
    OP_SHR,  
    
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,
    
    OP_LOAD_IMM,   
    OP_LOAD,     
    OP_STORE,      
    OP_LOAD_SHARED, 
    OP_STORE_SHARED, 
    
    OP_PUSH,
    OP_POP,
    OP_DUP,
    OP_SWAP,
    
    OP_JMP,        
    OP_JMP_TRUE,   
    OP_JMP_FALSE,  
    OP_CALL,
    OP_RET,
    
    OP_SYNC_WARP,  
    OP_SYNC_BLOCK, 
    
    OP_BALLOT,     
    OP_SHFL,       
    OP_SHFL_DOWN,  
    OP_SHFL_UP,    
    OP_ANY_SYNC,   
    OP_ALL_SYNC,   
    OP_REDUCE_ADD, 
    OP_REDUCE_MAX, 
    OP_REDUCE_MIN, 
    
    OP_CMOV,       
    OP_BRANCH_HINT, 
    
    OP_VLOAD2,     
    OP_VLOAD4,     
    OP_VSTORE2,    
    OP_VSTORE4,    
    
    OP_PREFETCH,   
    
    OP_LOAD_STRING_PTR,
    OP_STR_LEN,
    OP_STR_CMP,
    OP_STR_COPY,
    
    OP_MALLOC,
    OP_FREE,
    OP_PRINT,
    
    OP_HALT,
    OP_BREAKPOINT, 
    OP_MAX_OPCODE
} OpCode;

typedef struct {
    OpCode opcode;
    uint32_t operand;  
} Instruction;

typedef struct {
    Instruction* instructions;
    size_t instruction_count;
    size_t data_size;      
    uint8_t* data_segment; 
} BytecodeProgram;

BytecodeProgram* bytecode_create(size_t instruction_count, size_t data_size);
void bytecode_destroy(BytecodeProgram* program);
int bytecode_load_from_file(const char* filename, BytecodeProgram* program);
int bytecode_save_to_file(const char* filename, const BytecodeProgram* program);
void bytecode_print(const BytecodeProgram* program);

#ifdef __cplusplus
}
#endif

#endif