#ifndef GPU_VM_H
#define GPU_VM_H

#include "../vm/bytecode.h"
#include "gpu_memory.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    GPUMemory* gpu_mem;
    BytecodeProgram* program;
    size_t num_threads;
    size_t num_blocks;
    int device_id;
    int threads_per_block;      
    size_t memory_per_thread;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float kernel_time_ms;
    float transfer_time_ms;
    const char* last_error;
    void* d_thread_errors;     
    void* d_thread_instr;       
} GPUVirtualMachine;

GPUVirtualMachine* gpu_vm_create(size_t memory_size, size_t num_threads, int device_id);
void gpu_vm_destroy(GPUVirtualMachine* vm);
int gpu_vm_load_program(GPUVirtualMachine* vm, const BytecodeProgram* program);
int gpu_vm_execute(GPUVirtualMachine* vm, int max_iterations);
int gpu_vm_execute_async(GPUVirtualMachine* vm, int max_iterations);
int gpu_vm_wait(GPUVirtualMachine* vm);
int gpu_vm_get_results(GPUVirtualMachine* vm, int32_t* registers, int32_t* memory);
int gpu_vm_get_results_async(GPUVirtualMachine* vm, int32_t* registers, int32_t* memory);
int gpu_vm_set_thread_data(GPUVirtualMachine* vm, size_t thread_id, const int32_t* initial_registers, const int32_t* initial_memory);
int gpu_vm_get_performance_stats(const GPUVirtualMachine* vm, float* kernel_time, float* transfer_time);
const char* gpu_vm_get_last_error(const GPUVirtualMachine* vm);
void gpu_vm_print_info(const GPUVirtualMachine* vm);

int gpu_vm_set_memory_isolation(GPUVirtualMachine* vm, size_t memory_per_thread);
size_t gpu_vm_get_memory_per_thread(const GPUVirtualMachine* vm);

int gpu_vm_get_instrumentation(GPUVirtualMachine* vm, size_t thread_id, 
                                uint32_t* instruction_count, uint32_t* branch_count, 
                                uint32_t* memory_access_count);

#ifdef __cplusplus
}
#endif

#endif