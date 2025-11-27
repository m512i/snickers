#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "vm/bytecode.h"
#include "kernel/device_instructions.cuh"
#include "kernel/device_handlers_impl.cuh"

#define MAX_CONSTANT_PROGRAM_SIZE 8192 

__constant__ Instruction program_const[MAX_CONSTANT_PROGRAM_SIZE];

__device__ inline void prefetch_instruction_global(const Instruction* program, uint32_t next_pc, size_t program_size) {
    if (next_pc < program_size) {
        volatile const Instruction* prefetch_ptr = &program[next_pc];
        (void)*prefetch_ptr;  
    }
}

__device__ inline void warm_instruction_cache_global(const Instruction* program, uint32_t start_pc, size_t program_size) {
    const uint32_t warm_range = 32;
    volatile const Instruction* warm_ptr;
    for (uint32_t i = 0; i < warm_range && (start_pc + i) < program_size; i++) {
        warm_ptr = &program[start_pc + i];
        (void)*warm_ptr;  
    }
}

__device__ inline void warm_instruction_cache_const(uint32_t start_pc, size_t program_size) {
    const uint32_t warm_range = 32;
    volatile const Instruction* warm_ptr;
    for (uint32_t i = 0; i < warm_range && (start_pc + i) < program_size; i++) {
        warm_ptr = &program_const[start_pc + i];
        (void)*warm_ptr; 
    }
}

__device__ inline void prefetch_instruction_const(uint32_t next_pc, size_t program_size) {
    if (next_pc < program_size) {
        volatile const Instruction* prefetch_ptr = &program_const[next_pc];
        (void)*prefetch_ptr;
    }
}

typedef struct {
    uint32_t error_code;  
    uint32_t error_pc;   
} ThreadError;

typedef struct {
    uint32_t instruction_count;
    uint32_t branch_count;
    uint32_t memory_access_count;
} ThreadInstrumentation;

__global__ void vm_execute_kernel(
    const Instruction* program,
    size_t program_size,
    int32_t* global_memory,
    size_t memory_size,
    size_t memory_per_thread,  
    int32_t* output_registers,  
    uint32_t* program_counters,  
    ThreadError* thread_errors,  
    ThreadInstrumentation* thread_instr,  
    int max_iterations
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    DeviceVMState state;
    memset(&state, 0, sizeof(DeviceVMState));
    state.instruction_count = 0;
    state.branch_count = 0;
    state.memory_access_count = 0;
    
    uint32_t pc = program_counters[thread_id];
    if (pc >= program_size) {
        pc = 0;  
        if (thread_errors) {
            thread_errors[thread_id].error_code = 1; 
            thread_errors[thread_id].error_pc = pc;
        }
    }
    
    __shared__ int32_t shared_memory[1024];
    int32_t* thread_memory = global_memory;
    if (memory_per_thread > 0) {
        thread_memory = &global_memory[thread_id * memory_per_thread];
    }
    
    warm_instruction_cache_global(program, pc, program_size);
    
    int iterations = 0;
    while (iterations < max_iterations) {
        if (pc >= program_size) {
            if (thread_errors) {
                thread_errors[thread_id].error_code = 2;  
                thread_errors[thread_id].error_pc = pc;
            }
            break;  
        }
        
        const Instruction* instr = &program[pc];
        uint32_t next_pc = pc + 1;
        prefetch_instruction_global(program, next_pc, program_size);
        
        size_t effective_memory_size = (memory_per_thread > 0) ? memory_per_thread : memory_size;
        
        int halt = device_execute_instruction(
            instr,
            &state,
            shared_memory,
            &pc,
            thread_memory,  
            effective_memory_size,
            program_size  
        );
        
        if (halt != 0) {
            break;  
        }
        
        if (pc < program_size && pc != next_pc) {
            prefetch_instruction_global(program, pc, program_size);
        }
        
        iterations++;
        
        if (iterations >= max_iterations) {
            if (thread_errors) {
                thread_errors[thread_id].error_code = 3;  
                thread_errors[thread_id].error_pc = pc;
            }
            break;
        }
        
        if (pc >= program_size && pc != 0xFFFFFFFF) {
            if (thread_errors) {
                thread_errors[thread_id].error_code = 2;  
                thread_errors[thread_id].error_pc = pc;
            }
            break;
        }
    }
    
    program_counters[thread_id] = pc;
    
    int32_t* reg_base = &output_registers[thread_id * DEVICE_MAX_REGISTERS];
    #pragma unroll
    for (int i = 0; i < DEVICE_MAX_REGISTERS; i++) {
        reg_base[i] = state.registers[i];
    }
    
    if (thread_instr) {
        thread_instr[thread_id].instruction_count = state.instruction_count;
        thread_instr[thread_id].branch_count = state.branch_count;
        thread_instr[thread_id].memory_access_count = state.memory_access_count;
    }
}

__global__ void vm_execute_kernel_const(
    size_t program_size,
    int32_t* global_memory,
    size_t memory_size,
    size_t memory_per_thread, 
    int32_t* output_registers,
    uint32_t* program_counters,
    ThreadError* thread_errors,  
    ThreadInstrumentation* thread_instr,  
    int max_iterations
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    DeviceVMState state;
    memset(&state, 0, sizeof(DeviceVMState));
    state.instruction_count = 0;
    state.branch_count = 0;
    state.memory_access_count = 0;
    
    uint32_t pc = program_counters[thread_id];
    if (pc >= program_size) {
        pc = 0;  
        if (thread_errors) {
            thread_errors[thread_id].error_code = 1;  
            thread_errors[thread_id].error_pc = pc;
        }
    }
    
    __shared__ int32_t shared_memory[1024];
    int32_t* thread_memory = global_memory;
    if (memory_per_thread > 0) {
        thread_memory = &global_memory[thread_id * memory_per_thread];
    }
    
    warm_instruction_cache_const(pc, program_size);
    
    int iterations = 0;
    while (iterations < max_iterations) {
        if (pc >= program_size) {
            if (thread_errors) {
                thread_errors[thread_id].error_code = 2;  
                thread_errors[thread_id].error_pc = pc;
            }
            break;  
        }
        
        const Instruction* instr = &program_const[pc];
        uint32_t next_pc = pc + 1;
        prefetch_instruction_const(next_pc, program_size);
        
        size_t effective_memory_size = (memory_per_thread > 0) ? memory_per_thread : memory_size;
        
        int halt = device_execute_instruction(
            instr,
            &state,
            shared_memory,
            &pc,
            thread_memory,  
            effective_memory_size,
            program_size  
        );
        
        if (halt != 0) {
            break;
        }
        
        if (pc < program_size && pc != next_pc) {
            prefetch_instruction_const(pc, program_size);
        }
        
        iterations++;
        if (iterations >= max_iterations) {
            if (thread_errors) {
                thread_errors[thread_id].error_code = 3;  
                thread_errors[thread_id].error_pc = pc;
            }
            break;
        }
        
        if (pc >= program_size && pc != 0xFFFFFFFF) {
            if (thread_errors) {
                thread_errors[thread_id].error_code = 2;  
                thread_errors[thread_id].error_pc = pc;
            }
            break;
        }
    }
    
    program_counters[thread_id] = pc;
    int32_t* reg_base = &output_registers[thread_id * DEVICE_MAX_REGISTERS];
    #pragma unroll
    for (int i = 0; i < DEVICE_MAX_REGISTERS; i++) {
        reg_base[i] = state.registers[i];
    }
    if (thread_instr) {
        thread_instr[thread_id].instruction_count = state.instruction_count;
        thread_instr[thread_id].branch_count = state.branch_count;
        thread_instr[thread_id].memory_access_count = state.memory_access_count;
    }
}

extern "C" void launch_vm_kernel(
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
) {
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    if (stream) {
        vm_execute_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_program,
            program_size,
            d_memory,
            memory_size,
            memory_per_thread,
            d_registers,
            d_pc,
            d_thread_errors,
            d_thread_instr,
            max_iterations
        );
    } else {
        vm_execute_kernel<<<num_blocks, threads_per_block>>>(
            d_program,
            program_size,
            d_memory,
            memory_size,
            memory_per_thread,
            d_registers,
            d_pc,
            d_thread_errors,
            d_thread_instr,
            max_iterations
        );
    }
    
    if (!stream) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        }
    }
}

extern "C" void launch_vm_kernel_const(
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
) {
    cudaError_t err = cudaMemcpyToSymbol(program_const, h_program,
                                        sizeof(Instruction) * program_size,
                                        0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return;  
    }
    
    if (stream) {
        cudaStreamSynchronize(stream);
    }
    
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    
    if (stream) {
        vm_execute_kernel_const<<<num_blocks, threads_per_block, 0, stream>>>(
            program_size,
            d_memory,
            memory_size,
            memory_per_thread,
            d_registers,
            d_pc,
            d_thread_errors,
            d_thread_instr,
            max_iterations
        );
    } else {
        vm_execute_kernel_const<<<num_blocks, threads_per_block>>>(
            program_size,
            d_memory,
            memory_size,
            memory_per_thread,
            d_registers,
            d_pc,
            d_thread_errors,
            d_thread_instr,
            max_iterations
        );
    }
}

