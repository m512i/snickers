#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "../vm/bytecode.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t* d_memory;           
    int32_t* d_registers;        
    int32_t* d_stack;            
    uint32_t* d_pc;              
    uint32_t* d_flags;           
    Instruction* d_program;      
    int32_t* h_memory;
    int32_t* h_registers;
    size_t memory_size;
    size_t num_threads;
    size_t program_size;
    cudaStream_t stream;
    int use_pinned_memory;      
    int use_constant_memory;     
} GPUMemory;

GPUMemory* gpu_memory_create(size_t memory_size, size_t num_threads, size_t program_size);
void gpu_memory_destroy(GPUMemory* gpu_mem);
int gpu_memory_copy_to_device(GPUMemory* gpu_mem, const int32_t* host_memory, const Instruction* program);
int gpu_memory_copy_to_device_async(GPUMemory* gpu_mem, const int32_t* host_memory, const Instruction* program);
int gpu_memory_copy_from_device(GPUMemory* gpu_mem, int32_t* host_memory, int32_t* host_registers);
int gpu_memory_copy_from_device_async(GPUMemory* gpu_mem, int32_t* host_memory, int32_t* host_registers);
int gpu_memory_resize(GPUMemory* gpu_mem, size_t new_memory_size, size_t new_num_threads, size_t new_program_size);
const char* gpu_memory_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif