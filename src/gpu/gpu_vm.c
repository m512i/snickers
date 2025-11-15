#include "gpu/gpu_vm.h"
#include "gpu/gpu_memory.h"
#include "utils/colors.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t error_code;
    uint32_t error_pc;
} ThreadError;

typedef struct {
    uint32_t instruction_count;
    uint32_t branch_count;
    uint32_t memory_access_count;
} ThreadInstrumentation;

extern void launch_vm_kernel(
    const Instruction* d_program,
    size_t program_size,
    int32_t* d_memory,
    size_t memory_size,
    size_t memory_per_thread,
    int32_t* d_registers,
    uint32_t* d_pc,
    size_t num_threads,
    int max_iterations,
    int threads_per_block,
    cudaStream_t stream,
    ThreadError* d_thread_errors,
    ThreadInstrumentation* d_thread_instr
);

extern void launch_vm_kernel_const(
    const Instruction* h_program,
    size_t program_size,
    int32_t* d_memory,
    size_t memory_size,
    size_t memory_per_thread,
    int32_t* d_registers,
    uint32_t* d_pc,
    size_t num_threads,
    int max_iterations,
    int threads_per_block,
    cudaStream_t stream,
    ThreadError* d_thread_errors,
    ThreadInstrumentation* d_thread_instr
);


static int get_optimal_block_size(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return 256;  
    }
    int max_threads = prop.maxThreadsPerBlock;
    if (max_threads > 1024) max_threads = 1024;
    
    int threads = 256;
    while (threads * 2 <= max_threads) {
        threads *= 2;
    }
    
    return threads;
}

GPUVirtualMachine* gpu_vm_create(size_t memory_size, size_t num_threads, int device_id) {
    GPUVirtualMachine* vm = (GPUVirtualMachine*)malloc(sizeof(GPUVirtualMachine));
    if (!vm) return NULL;
    
    memset(vm, 0, sizeof(GPUVirtualMachine));
    
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        free(vm);
        return NULL;
    }
    
    vm->device_id = device_id;
    vm->num_threads = num_threads;
    vm->memory_per_thread = 0;
    
    vm->threads_per_block = get_optimal_block_size(device_id);
    vm->num_blocks = (num_threads + vm->threads_per_block - 1) / vm->threads_per_block;
    
    err = cudaEventCreate(&vm->start_event);
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        free(vm);
        return NULL;
    }
    
    err = cudaEventCreate(&vm->stop_event);
    if (err != cudaSuccess) {
        cudaEventDestroy(vm->start_event);
        vm->last_error = cudaGetErrorString(err);
        free(vm);
        return NULL;
    }
    
    vm->gpu_mem = gpu_memory_create(memory_size, num_threads, 0);
    if (!vm->gpu_mem) {
        cudaEventDestroy(vm->stop_event);
        cudaEventDestroy(vm->start_event);
        vm->last_error = "Failed to create GPU memory";
        free(vm);
        return NULL;
    }
    
    vm->program = NULL;
    vm->last_error = NULL;
    vm->kernel_time_ms = 0.0f;
    vm->transfer_time_ms = 0.0f;
    
    return vm;
}

void gpu_vm_destroy(GPUVirtualMachine* vm) {
    if (!vm) return;
    
    if (vm->gpu_mem) {
        gpu_memory_destroy(vm->gpu_mem);
    }
    
    if (vm->d_thread_errors) {
        cudaFree(vm->d_thread_errors);
    }
    if (vm->d_thread_instr) {
        cudaFree(vm->d_thread_instr);
    }
    
    if (vm->start_event) {
        cudaEventDestroy(vm->start_event);
    }
    
    if (vm->stop_event) {
        cudaEventDestroy(vm->stop_event);
    }
    
    free(vm);
}

int gpu_vm_load_program(GPUVirtualMachine* vm, const BytecodeProgram* program) {
    if (!vm || !program) {
        if (vm) vm->last_error = "Invalid arguments";
        return -1;
    }
    
    vm->program = (BytecodeProgram*)program;  
    int needs_resize = 0;
    
    if (!vm->gpu_mem) {
        needs_resize = 1;
    } else if (vm->gpu_mem->program_size != program->instruction_count) {
        needs_resize = 1;
    }
    
    if (needs_resize) {
        size_t old_memory_size = vm->gpu_mem ? vm->gpu_mem->memory_size : 0;
        size_t old_num_threads = vm->num_threads;
        
        if (vm->gpu_mem) {
            gpu_memory_destroy(vm->gpu_mem);
        }
        
        vm->gpu_mem = gpu_memory_create(
            old_memory_size > 0 ? old_memory_size : 65536,
            old_num_threads > 0 ? old_num_threads : vm->num_threads,
            program->instruction_count
        );
        if (!vm->gpu_mem) {
            vm->last_error = "Failed to create GPU memory for program";
            return -2;
        }
    } else {
        vm->gpu_mem->program_size = program->instruction_count;
        vm->gpu_mem->use_constant_memory = (program->instruction_count > 0 && program->instruction_count <= 8192) ? 1 : 0;
    }
    
    int32_t* initial_memory = NULL;
    if (vm->gpu_mem->use_pinned_memory) {
        initial_memory = vm->gpu_mem->h_memory;
        memset(initial_memory, 0, sizeof(int32_t) * vm->gpu_mem->memory_size);
    } else {
        initial_memory = (int32_t*)calloc(vm->gpu_mem->memory_size, sizeof(int32_t));
        if (!initial_memory) {
            vm->last_error = "Failed to allocate initial memory";
            return -3;
        }
    }
    
    if (program->data_segment && program->data_size > 0) {
        size_t copy_size = (program->data_size < vm->gpu_mem->memory_size * sizeof(int32_t)) ?
                          program->data_size : vm->gpu_mem->memory_size * sizeof(int32_t);
        memcpy(initial_memory, program->data_segment, copy_size);
    }
    
    cudaEventRecord(vm->start_event, vm->gpu_mem->stream);
    int result = gpu_memory_copy_to_device_async(vm->gpu_mem, initial_memory, program->instructions);
    cudaEventRecord(vm->stop_event, vm->gpu_mem->stream);
    
    if (!vm->gpu_mem->use_pinned_memory) {
        free(initial_memory);
    }
    
    if (result != 0) {
        vm->last_error = gpu_memory_get_error_string(result);
        return -4;
    }
    
    cudaError_t err = cudaStreamSynchronize(vm->gpu_mem->stream);
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        return -5;
    }
    
    cudaEventElapsedTime(&vm->transfer_time_ms, vm->start_event, vm->stop_event);
    
    return 0;
}

int gpu_vm_execute(GPUVirtualMachine* vm, int max_iterations) {
    if (!vm || !vm->program) {
        if (vm) vm->last_error = "Invalid VM or no program loaded";
        return -1;
    }
    
    cudaEventRecord(vm->start_event, vm->gpu_mem->stream);
    
    if (vm->gpu_mem->use_constant_memory) {
        launch_vm_kernel_const(
            vm->program->instructions,  
            vm->program->instruction_count,
            vm->gpu_mem->d_memory,
            vm->gpu_mem->memory_size,
            vm->memory_per_thread,  
            vm->gpu_mem->d_registers,
            vm->gpu_mem->d_pc,
            vm->num_threads,
            max_iterations,
            vm->threads_per_block,
            vm->gpu_mem->stream,
            (ThreadError*)vm->d_thread_errors,  
            (ThreadInstrumentation*)vm->d_thread_instr  
        );
    } else {
        launch_vm_kernel(
            vm->gpu_mem->d_program,
            vm->program->instruction_count,
            vm->gpu_mem->d_memory,
            vm->gpu_mem->memory_size,
            vm->memory_per_thread,  
            vm->gpu_mem->d_registers,
            vm->gpu_mem->d_pc,
            vm->num_threads,
            max_iterations,
            vm->threads_per_block,
            vm->gpu_mem->stream,
            (ThreadError*)vm->d_thread_errors, 
            (ThreadInstrumentation*)vm->d_thread_instr  
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        return -2;
    }
    
    err = cudaStreamSynchronize(vm->gpu_mem->stream);
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        return -3;
    }
    
    cudaEventRecord(vm->stop_event, vm->gpu_mem->stream);
    cudaStreamSynchronize(vm->gpu_mem->stream);
    cudaEventElapsedTime(&vm->kernel_time_ms, vm->start_event, vm->stop_event);
    
    return 0;
}

int gpu_vm_execute_async(GPUVirtualMachine* vm, int max_iterations) {
    if (!vm || !vm->program) {
        if (vm) vm->last_error = "Invalid VM or no program loaded";
        return -1;
    }
    
    cudaEventRecord(vm->start_event, vm->gpu_mem->stream);
    
    if (vm->gpu_mem->use_constant_memory) {
        launch_vm_kernel_const(
            vm->program->instructions,
            vm->program->instruction_count,
            vm->gpu_mem->d_memory,
            vm->gpu_mem->memory_size,
            vm->memory_per_thread,  
            vm->gpu_mem->d_registers,
            vm->gpu_mem->d_pc,
            vm->num_threads,
            max_iterations,
            vm->threads_per_block,
            vm->gpu_mem->stream,
            (ThreadError*)vm->d_thread_errors,  
            (ThreadInstrumentation*)vm->d_thread_instr  
        );
    } else {
        launch_vm_kernel(
            vm->gpu_mem->d_program,
            vm->program->instruction_count,
            vm->gpu_mem->d_memory,
            vm->gpu_mem->memory_size,
            vm->memory_per_thread,  
            vm->gpu_mem->d_registers,
            vm->gpu_mem->d_pc,
            vm->num_threads,
            max_iterations,
            vm->threads_per_block,
            vm->gpu_mem->stream,
            (ThreadError*)vm->d_thread_errors,  
            (ThreadInstrumentation*)vm->d_thread_instr  
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        return -2;
    }
    return 0;
}

int gpu_vm_wait(GPUVirtualMachine* vm) {
    if (!vm) return -1;
    
    cudaError_t err = cudaStreamSynchronize(vm->gpu_mem->stream);
    if (err != cudaSuccess) {
        vm->last_error = cudaGetErrorString(err);
        return -1;
    }
    
    cudaEventRecord(vm->stop_event, vm->gpu_mem->stream);
    cudaStreamSynchronize(vm->gpu_mem->stream);
    cudaEventElapsedTime(&vm->kernel_time_ms, vm->start_event, vm->stop_event);
    
    return 0;
}

int gpu_vm_get_results(GPUVirtualMachine* vm, int32_t* registers, int32_t* memory) {
    if (!vm) return -1;
    
    if (vm->gpu_mem->use_pinned_memory) {
        int result = gpu_memory_copy_from_device_async(
            vm->gpu_mem,
            memory ? vm->gpu_mem->h_memory : NULL,
            registers ? vm->gpu_mem->h_registers : NULL
        );
        if (result != 0) {
            vm->last_error = gpu_memory_get_error_string(result);
            return result;
        }
        
        cudaError_t err = cudaStreamSynchronize(vm->gpu_mem->stream);
        if (err != cudaSuccess) {
            vm->last_error = cudaGetErrorString(err);
            return -2;
        }
        
        if (memory) {
            memcpy(memory, vm->gpu_mem->h_memory, sizeof(int32_t) * vm->gpu_mem->memory_size);
        }
        if (registers) {
            memcpy(registers, vm->gpu_mem->h_registers, 
                   sizeof(int32_t) * vm->gpu_mem->num_threads * 32);
        }
    } else {
        int result = gpu_memory_copy_from_device(vm->gpu_mem, memory, registers);
        if (result != 0) {
            vm->last_error = gpu_memory_get_error_string(result);
            return result;
        }
    }
    
    return 0;
}

int gpu_vm_get_results_async(GPUVirtualMachine* vm, int32_t* registers, int32_t* memory) {
    if (!vm) return -1;
    
    int result = gpu_memory_copy_from_device_async(vm->gpu_mem, memory, registers);
    if (result != 0) {
        vm->last_error = gpu_memory_get_error_string(result);
        return result;
    }
    
    return 0;
}

int gpu_vm_set_thread_data(GPUVirtualMachine* vm, size_t thread_id, 
                           const int32_t* initial_registers, const int32_t* initial_memory) {
    if (!vm || thread_id >= vm->num_threads) {
        if (vm) vm->last_error = "Invalid thread ID";
        return -1;
    }
    
    if (initial_registers) {
        cudaError_t err = cudaMemcpyAsync(
            &vm->gpu_mem->d_registers[thread_id * 32],
            initial_registers,
            sizeof(int32_t) * 32,
            cudaMemcpyHostToDevice,
            vm->gpu_mem->stream
        );
        if (err != cudaSuccess) {
            vm->last_error = cudaGetErrorString(err);
            return -2;
        }
    }
    
    if (initial_memory) {
        // For now, only copy to thread 0's memory region
        // Future: Support per-thread memory regions
        if (thread_id == 0) {
            cudaError_t err = cudaMemcpyAsync(
                vm->gpu_mem->d_memory,
                initial_memory,
                sizeof(int32_t) * vm->gpu_mem->memory_size,
                cudaMemcpyHostToDevice,
                vm->gpu_mem->stream
            );
            if (err != cudaSuccess) {
                vm->last_error = cudaGetErrorString(err);
                return -3;
            }
        }
    }
    
    return 0;
}

int gpu_vm_get_performance_stats(const GPUVirtualMachine* vm, float* kernel_time, float* transfer_time) {
    if (!vm) return -1;
    
    if (kernel_time) {
        *kernel_time = vm->kernel_time_ms;
    }
    if (transfer_time) {
        *transfer_time = vm->transfer_time_ms;
    }
    
    return 0;
}

const char* gpu_vm_get_last_error(const GPUVirtualMachine* vm) {
    if (!vm) return "Invalid VM";
    return vm->last_error ? vm->last_error : "No error";
}

void gpu_vm_print_info(const GPUVirtualMachine* vm) {
    if (!vm) return;
    
    printf(COLOR_HEADER "GPU VM Info:" COLOR_RESET "\n");
    printf("  " COLOR_LABEL "Device ID:" COLOR_RESET " " COLOR_VALUE "%d" COLOR_RESET "\n", vm->device_id);
    printf("  " COLOR_LABEL "Threads:" COLOR_RESET " " COLOR_VALUE "%zu" COLOR_RESET "\n", vm->num_threads);
    printf("  " COLOR_LABEL "Blocks:" COLOR_RESET " " COLOR_VALUE "%zu" COLOR_RESET "\n", vm->num_blocks);
    printf("  " COLOR_LABEL "Threads per Block:" COLOR_RESET " " COLOR_VALUE "%d" COLOR_RESET "\n", vm->threads_per_block);
    printf("  " COLOR_LABEL "Memory Size:" COLOR_RESET " " COLOR_VALUE "%zu bytes" COLOR_RESET "\n", vm->gpu_mem->memory_size);
    printf("  " COLOR_LABEL "Pinned Memory:" COLOR_RESET " %s\n", 
           vm->gpu_mem->use_pinned_memory ? COLOR_SUCCESS "Yes" : COLOR_WARNING "No");
    printf("  " COLOR_LABEL "Constant Memory:" COLOR_RESET " %s\n", 
           vm->gpu_mem->use_constant_memory ? COLOR_SUCCESS "Yes" : COLOR_WARNING "No");
    if (vm->program) {
        printf("  " COLOR_LABEL "Program Size:" COLOR_RESET " " COLOR_VALUE "%zu instructions" COLOR_RESET "\n", vm->program->instruction_count);
    }
    if (vm->kernel_time_ms > 0) {
        printf("  " COLOR_LABEL "Last Kernel Time:" COLOR_RESET " " COLOR_VALUE "%.3f ms" COLOR_RESET "\n", vm->kernel_time_ms);
    }
    if (vm->transfer_time_ms > 0) {
        printf("  " COLOR_LABEL "Last Transfer Time:" COLOR_RESET " " COLOR_VALUE "%.3f ms" COLOR_RESET "\n", vm->transfer_time_ms);
    }
    if (vm->memory_per_thread > 0) {
        printf("  " COLOR_LABEL "Memory per Thread:" COLOR_RESET " " COLOR_VALUE "%zu words" COLOR_METADATA " (isolated)" COLOR_RESET "\n", vm->memory_per_thread);
    } else {
        printf("  " COLOR_LABEL "Memory:" COLOR_RESET " " COLOR_INFO "Shared (all threads)" COLOR_RESET "\n");
    }
    printf(COLOR_RESET);
}

int gpu_vm_set_memory_isolation(GPUVirtualMachine* vm, size_t memory_per_thread) {
    if (!vm) return -1;
    
    size_t total_required = memory_per_thread * vm->num_threads;
    if (memory_per_thread > 0 && total_required > vm->gpu_mem->memory_size) {
        vm->last_error = "Insufficient memory for per-thread isolation";
        return -2;
    }
    
    vm->memory_per_thread = memory_per_thread;
    return 0;
}

size_t gpu_vm_get_memory_per_thread(const GPUVirtualMachine* vm) {
    if (!vm) return 0;
    return vm->memory_per_thread;
}

int gpu_vm_get_instrumentation(GPUVirtualMachine* vm, size_t thread_id,
                                uint32_t* instruction_count, uint32_t* branch_count,
                                uint32_t* memory_access_count) {
    if (!vm || !vm->d_thread_instr || thread_id >= vm->num_threads) {
        if (vm) vm->last_error = "Invalid parameters or instrumentation not enabled";
        return -1;
    }
    
    ThreadInstrumentation* h_instr = (ThreadInstrumentation*)malloc(sizeof(ThreadInstrumentation));
    if (!h_instr) {
        vm->last_error = "Failed to allocate instrumentation buffer";
        return -2;
    }
    
    cudaError_t err = cudaMemcpy(h_instr, 
                                 &((ThreadInstrumentation*)vm->d_thread_instr)[thread_id],
                                 sizeof(ThreadInstrumentation),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_instr);
        vm->last_error = cudaGetErrorString(err);
        return -3;
    }
    
    if (instruction_count) *instruction_count = h_instr->instruction_count;
    if (branch_count) *branch_count = h_instr->branch_count;
    if (memory_access_count) *memory_access_count = h_instr->memory_access_count;
    
    free(h_instr);
    return 0;
}

#ifdef __cplusplus
}
#endif