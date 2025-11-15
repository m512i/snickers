#include "gpu/gpu_memory.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>

static const char* error_strings[] = {
    "Success",
    "Invalid argument",
    "Memory copy to device failed",
    "Program copy to device failed",
    "Memory copy from device failed",
    "Registers copy from device failed",
    "CUDA stream creation failed",
    "Pinned memory allocation failed",
    "Memory resize failed"
};

const char* gpu_memory_get_error_string(int error_code) {
    if (error_code < 0 || error_code >= (int)(sizeof(error_strings) / sizeof(error_strings[0]))) {
        return "Unknown error";
    }
    return error_strings[-error_code];
}

GPUMemory* gpu_memory_create(size_t memory_size, size_t num_threads, size_t program_size) {
    GPUMemory* gpu_mem = (GPUMemory*)malloc(sizeof(GPUMemory));
    if (!gpu_mem) return NULL;
    
    memset(gpu_mem, 0, sizeof(GPUMemory));
    gpu_mem->memory_size = memory_size;
    gpu_mem->num_threads = num_threads;
    gpu_mem->program_size = program_size;
    gpu_mem->use_pinned_memory = 1;
    gpu_mem->use_constant_memory = (program_size > 0 && program_size <= 8192) ? 1 : 0;
    
    cudaError_t err;
    
    err = cudaStreamCreate(&gpu_mem->stream);
    if (err != cudaSuccess) {
        free(gpu_mem);
        return NULL;
    }
    
    err = cudaMalloc((void**)&gpu_mem->d_memory, sizeof(int32_t) * memory_size);
    if (err != cudaSuccess) {
        cudaStreamDestroy(gpu_mem->stream);
        free(gpu_mem);
        return NULL;
    }
    
    err = cudaMalloc((void**)&gpu_mem->d_registers, sizeof(int32_t) * num_threads * 32);
    if (err != cudaSuccess) {
        cudaFree(gpu_mem->d_memory);
        cudaStreamDestroy(gpu_mem->stream);
        free(gpu_mem);
        return NULL;
    }
    
    err = cudaMalloc((void**)&gpu_mem->d_stack, sizeof(int32_t) * num_threads * 256);
    if (err != cudaSuccess) {
        cudaFree(gpu_mem->d_registers);
        cudaFree(gpu_mem->d_memory);
        cudaStreamDestroy(gpu_mem->stream);
        free(gpu_mem);
        return NULL;
    }
    
    err = cudaMalloc((void**)&gpu_mem->d_pc, sizeof(uint32_t) * num_threads);
    if (err != cudaSuccess) {
        cudaFree(gpu_mem->d_stack);
        cudaFree(gpu_mem->d_registers);
        cudaFree(gpu_mem->d_memory);
        cudaStreamDestroy(gpu_mem->stream);
        free(gpu_mem);
        return NULL;
    }
    
    err = cudaMalloc((void**)&gpu_mem->d_flags, sizeof(uint32_t) * num_threads);
    if (err != cudaSuccess) {
        cudaFree(gpu_mem->d_pc);
        cudaFree(gpu_mem->d_stack);
        cudaFree(gpu_mem->d_registers);
        cudaFree(gpu_mem->d_memory);
        cudaStreamDestroy(gpu_mem->stream);
        free(gpu_mem);
        return NULL;
    }
    
    if (!gpu_mem->use_constant_memory && program_size > 0) {
        err = cudaMalloc((void**)&gpu_mem->d_program, sizeof(Instruction) * program_size);
        if (err != cudaSuccess) {
            cudaFree(gpu_mem->d_flags);
            cudaFree(gpu_mem->d_pc);
            cudaFree(gpu_mem->d_stack);
            cudaFree(gpu_mem->d_registers);
            cudaFree(gpu_mem->d_memory);
            cudaStreamDestroy(gpu_mem->stream);
            free(gpu_mem);
            return NULL;
        }
    } else {
        gpu_mem->d_program = NULL;
    }
    
    if (gpu_mem->use_pinned_memory) {
        err = cudaMallocHost((void**)&gpu_mem->h_memory, sizeof(int32_t) * memory_size);
        if (err != cudaSuccess) {
            gpu_mem->h_memory = (int32_t*)malloc(sizeof(int32_t) * memory_size);
            if (!gpu_mem->h_memory) {
                gpu_memory_destroy(gpu_mem);
                return NULL;
            }
            gpu_mem->use_pinned_memory = 0;
        }
        
        err = cudaMallocHost((void**)&gpu_mem->h_registers, sizeof(int32_t) * num_threads * 32);
        if (err != cudaSuccess) {
            if (gpu_mem->use_pinned_memory) {
                cudaFreeHost(gpu_mem->h_memory);
            } else {
                free(gpu_mem->h_memory);
            }
            gpu_mem->h_registers = (int32_t*)malloc(sizeof(int32_t) * num_threads * 32);
            if (!gpu_mem->h_registers) {
                gpu_memory_destroy(gpu_mem);
                return NULL;
            }
        }
    } else {
        gpu_mem->h_memory = (int32_t*)malloc(sizeof(int32_t) * memory_size);
        if (!gpu_mem->h_memory) {
            gpu_memory_destroy(gpu_mem);
            return NULL;
        }
        
        gpu_mem->h_registers = (int32_t*)malloc(sizeof(int32_t) * num_threads * 32);
        if (!gpu_mem->h_registers) {
            gpu_memory_destroy(gpu_mem);
            return NULL;
        }
    }
    
    cudaMemsetAsync(gpu_mem->d_memory, 0, sizeof(int32_t) * memory_size, gpu_mem->stream);
    cudaMemsetAsync(gpu_mem->d_registers, 0, sizeof(int32_t) * num_threads * 32, gpu_mem->stream);
    cudaMemsetAsync(gpu_mem->d_stack, 0, sizeof(int32_t) * num_threads * 256, gpu_mem->stream);
    cudaMemsetAsync(gpu_mem->d_pc, 0, sizeof(uint32_t) * num_threads, gpu_mem->stream);
    cudaMemsetAsync(gpu_mem->d_flags, 0, sizeof(uint32_t) * num_threads, gpu_mem->stream);
    
    return gpu_mem;
}

void gpu_memory_destroy(GPUMemory* gpu_mem) {
    if (!gpu_mem) return;
    
    if (gpu_mem->stream) {
        cudaStreamDestroy(gpu_mem->stream);
    }
    
    if (gpu_mem->d_memory) cudaFree(gpu_mem->d_memory);
    if (gpu_mem->d_registers) cudaFree(gpu_mem->d_registers);
    if (gpu_mem->d_stack) cudaFree(gpu_mem->d_stack);
    if (gpu_mem->d_pc) cudaFree(gpu_mem->d_pc);
    if (gpu_mem->d_flags) cudaFree(gpu_mem->d_flags);
    if (gpu_mem->d_program) cudaFree(gpu_mem->d_program);
    
    if (gpu_mem->h_memory) {
        if (gpu_mem->use_pinned_memory) {
            cudaFreeHost(gpu_mem->h_memory);
        } else {
            free(gpu_mem->h_memory);
        }
    }
    if (gpu_mem->h_registers) {
        if (gpu_mem->use_pinned_memory) {
            cudaFreeHost(gpu_mem->h_registers);
        } else {
            free(gpu_mem->h_registers);
        }
    }
    
    free(gpu_mem);
}

int gpu_memory_copy_to_device(GPUMemory* gpu_mem, const int32_t* host_memory, const Instruction* program) {
    if (!gpu_mem) return -1;
    
    cudaError_t err;
    
    if (host_memory) {
        err = cudaMemcpy(gpu_mem->d_memory, host_memory, 
                        sizeof(int32_t) * gpu_mem->memory_size, 
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -2;
    }
    
    if (program) {
        if (gpu_mem->use_constant_memory) {
            if (gpu_mem->program_size > 8192) {
                return -3;
            }
        } else {
            err = cudaMemcpy(gpu_mem->d_program, program,
                            sizeof(Instruction) * gpu_mem->program_size,
                            cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return -3;
        }
    }
    
    return 0;
}

int gpu_memory_copy_to_device_async(GPUMemory* gpu_mem, const int32_t* host_memory, const Instruction* program) {
    if (!gpu_mem) return -1;
    
    cudaError_t err;
    
    if (host_memory) {
        err = cudaMemcpyAsync(gpu_mem->d_memory, host_memory, 
                             sizeof(int32_t) * gpu_mem->memory_size, 
                             cudaMemcpyHostToDevice, gpu_mem->stream);
        if (err != cudaSuccess) return -2;
    }
    
    if (program) {
        if (gpu_mem->use_constant_memory) {
            /* Constant memory is copied through launch_vm_kernel_const function
             For async copy, we just store the program pointer and let the kernel
             launch function handle it. For now, return success.
             The actual copy happens in launch_vm_kernel_const
             This is a limitation - constant memory must be copied synchronously
             in the kernel launch function, not here */
            (void)program;
        } else {
            err = cudaMemcpyAsync(gpu_mem->d_program, program,
                                 sizeof(Instruction) * gpu_mem->program_size,
                                 cudaMemcpyHostToDevice, gpu_mem->stream);
            if (err != cudaSuccess) return -3;
        }
    }
    
    return 0;
}

int gpu_memory_copy_from_device(GPUMemory* gpu_mem, int32_t* host_memory, int32_t* host_registers) {
    if (!gpu_mem) return -1;
    
    cudaError_t err;
    
    if (host_memory) {
        err = cudaMemcpy(host_memory, gpu_mem->d_memory,
                        sizeof(int32_t) * gpu_mem->memory_size,
                        cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -2;
    }
    
    if (host_registers) {
        err = cudaMemcpy(host_registers, gpu_mem->d_registers,
                        sizeof(int32_t) * gpu_mem->num_threads * 32,
                        cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -3;
    }
    
    return 0;
}

int gpu_memory_copy_from_device_async(GPUMemory* gpu_mem, int32_t* host_memory, int32_t* host_registers) {
    if (!gpu_mem) return -1;
    
    cudaError_t err;
    
    if (host_memory) {
        err = cudaMemcpyAsync(host_memory, gpu_mem->d_memory,
                             sizeof(int32_t) * gpu_mem->memory_size,
                             cudaMemcpyDeviceToHost, gpu_mem->stream);
        if (err != cudaSuccess) return -2;
    }
    
    if (host_registers) {
        err = cudaMemcpyAsync(host_registers, gpu_mem->d_registers,
                             sizeof(int32_t) * gpu_mem->num_threads * 32,
                             cudaMemcpyDeviceToHost, gpu_mem->stream);
        if (err != cudaSuccess) return -3;
    }
    
    return 0;
}

int gpu_memory_resize(GPUMemory* gpu_mem, size_t new_memory_size, size_t new_num_threads, size_t new_program_size) {
    if (!gpu_mem) return -1;

    if (gpu_mem->memory_size == new_memory_size &&
        gpu_mem->num_threads == new_num_threads &&
        gpu_mem->program_size == new_program_size) {
        return 0;
    }
    
    /* For now, return error - proper resize would require reallocation
     This is a placeholder for future optimization
     In practice, we avoid destroying/recreating in gpu_vm_load_program */
    return -1;
}