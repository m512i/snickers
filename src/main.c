#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "vm/bytecode.h"
#include "vm/vm_state.h"
#include "gpu/gpu_vm.h"
#include "utils/dump.h"
#include "utils/timer.h"
#include "utils/logger.h"

#ifdef __cplusplus
inline Instruction make_instruction(OpCode op, uint32_t operand) {
    Instruction instr;
    instr.opcode = op;
    instr.operand = operand;
    return instr;
}
#else
static inline Instruction make_instruction(OpCode op, uint32_t operand) {
    Instruction instr;
    instr.opcode = op;
    instr.operand = operand;
    return instr;
}
#endif

int main(int argc, char** argv) {
    logger_init();
    
    LOG_COLOR(COLOR_TITLE, "GPU Bytecode VM\n");
    LOG_COLOR(COLOR_TITLE, "===============\n\n");
    
    size_t num_threads = 1024;
    size_t memory_size = 65536;  
    int max_iterations = 10000;
    int device_id = 0;
    const char* program_file = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            memory_size = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            max_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            program_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            LOG_COLOR(COLOR_TITLE, "Usage: %s [options]\n", argv[0]);
            LOG_COLOR(COLOR_LABEL, "Options:\n");
            LOG_COLOR(COLOR_INFO, "  -t <num>    Number of threads (default: 1024)\n");
            LOG_COLOR(COLOR_INFO, "  -m <size>   Memory size in bytes (default: 65536)\n");
            LOG_COLOR(COLOR_INFO, "  -i <num>    Max iterations per thread (default: 10000)\n");
            LOG_COLOR(COLOR_INFO, "  -d <id>     CUDA device ID (default: 0)\n");
            LOG_COLOR(COLOR_INFO, "  -f <file>   Bytecode program file\n");
            LOG_COLOR(COLOR_INFO, "  -h          Show this help\n");
            return 0;
        }
    }
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        LOG_ERROR("Error: No CUDA devices found or CUDA not available.\n");
        LOG_ERROR("Make sure you have an NVIDIA GPU with CUDA support installed.\n");
        return 1;
    }
    
    LOG_COLOR(COLOR_INFO, "Found ");
    LOG_COLOR(COLOR_VALUE, "%d", device_count);
    LOG_COLOR(COLOR_INFO, " CUDA device(s)\n");
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaError_t prop_err = cudaGetDeviceProperties(&prop, i);
        if (prop_err == cudaSuccess) {
            LOG_COLOR(COLOR_INFO, "  Device %d: ", i);
            LOG_COLOR(COLOR_VALUE, "%s", prop.name);
            LOG_COLOR(COLOR_METADATA, " (Compute %d.%d)", prop.major, prop.minor);
            if (i == device_id) {
                LOG_COLOR(COLOR_SUCCESS, " [SELECTED]\n");
            } else {
                LOG_COLOR(COLOR_RESET, "\n");
            }
        }
    }
    
    if (device_id < 0 || device_id >= device_count) {
        LOG_ERROR("Error: Invalid device ID %d. Valid range: 0-%d\n", device_id, device_count - 1);
        return 1;
    }
    
    LOG_COLOR(COLOR_INFO, "\nUsing device %d\n", device_id);
    
    GPUVirtualMachine* vm = gpu_vm_create(memory_size, num_threads, device_id);
    if (!vm) {
        LOG_ERROR("Error: Failed to create GPU VM\n");
        return 1;
    }
    
    gpu_vm_print_info(vm);
    LOG_COLOR(COLOR_RESET, "\n");
    
    BytecodeProgram* program = NULL;
    
    const char* default_file = "gpu_tests/add_data.bc";
    const char* file_to_load = program_file ? program_file : default_file;
    
    LOG_COLOR(COLOR_INFO, "Loading program from file: ");
    LOG_COLOR(COLOR_VALUE, "%s\n", file_to_load);
    BytecodeProgram temp_program = {0};
    int result = bytecode_load_from_file(file_to_load, &temp_program);
    if (result != 0) {
        LOG_ERROR("Error: Failed to load program from file: %s (error code: %d)\n", file_to_load, result);
        LOG_ERROR("Make sure the file exists. You can generate test files by running generate_gpu_tests.exe\n");
        gpu_vm_destroy(vm);
        return 1;
    }
    program = bytecode_create(temp_program.instruction_count, temp_program.data_size);
    if (!program) {
        LOG_ERROR("Error: Failed to allocate program memory\n");
        gpu_vm_destroy(vm);
        return 1;
    }
    memcpy(program->instructions, temp_program.instructions, 
           sizeof(Instruction) * temp_program.instruction_count);
    if (temp_program.data_size > 0 && temp_program.data_segment) {
        memcpy(program->data_segment, temp_program.data_segment, temp_program.data_size);
    }
    free(temp_program.instructions);
    if (temp_program.data_segment) free(temp_program.data_segment);
    
    if (!program) {
        LOG_ERROR("Error: Failed to create program\n");
        gpu_vm_destroy(vm);
        return 1;
    }
    
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_COLOR(COLOR_HEADER, "Program:\n");
    dump_program(program);
    
    if (program->data_size > 0 && program->data_segment) {
        LOG_COLOR(COLOR_INFO, "\nData segment will initialize GPU memory[0-%zu] with:\n", 
                  (program->data_size / sizeof(int32_t)) - 1);
        int32_t* data = (int32_t*)program->data_segment;
        for (size_t i = 0; i < program->data_size / sizeof(int32_t); i++) {
            LOG_COLOR(COLOR_INFO, "  memory[%zu] = ", i);
            LOG_COLOR(COLOR_VALUE, "%d\n", data[i]);
        }
    }
    
    if (gpu_vm_load_program(vm, program) != 0) {
        const char* error = gpu_vm_get_last_error(vm);
        LOG_ERROR("Error: Failed to load program: %s\n", error);
        bytecode_destroy(program);
        gpu_vm_destroy(vm);
        return 1;
    }
    
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_COLOR(COLOR_HEADER, "GPU VM Info (after loading program):\n");
    gpu_vm_print_info(vm);
    LOG_COLOR(COLOR_RESET, "\n");
    
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_INFO("Executing on GPU...\n");
    Timer timer;
    timer_start(&timer);
    
    if (gpu_vm_execute(vm, max_iterations) != 0) {
        const char* error = gpu_vm_get_last_error(vm);
        LOG_ERROR("Error: GPU execution failed: %s\n", error);
        bytecode_destroy(program);
        gpu_vm_destroy(vm);
        return 1;
    }
    
    timer_stop(&timer);
    
    float kernel_time = 0.0f, transfer_time = 0.0f;
    gpu_vm_get_performance_stats(vm, &kernel_time, &transfer_time);
    
    LOG_COLOR(COLOR_LABEL, "Execution time: ");
    LOG_COLOR(COLOR_VALUE, "%.3f ms", timer_elapsed_ms(&timer));
    LOG_COLOR(COLOR_METADATA, " (Host timer)\n");
    if (kernel_time > 0) {
        LOG_COLOR(COLOR_LABEL, "Kernel execution: ");
        LOG_COLOR(COLOR_VALUE, "%.3f ms", kernel_time);
        LOG_COLOR(COLOR_METADATA, " (CUDA event)\n");
    }
    if (transfer_time > 0) {
        LOG_COLOR(COLOR_LABEL, "Memory transfer: ");
        LOG_COLOR(COLOR_VALUE, "%.3f ms", transfer_time);
        LOG_COLOR(COLOR_METADATA, " (CUDA event)\n");
    }
    
    int32_t* registers = (int32_t*)malloc(sizeof(int32_t) * num_threads * 32);
    int32_t* memory = (int32_t*)malloc(sizeof(int32_t) * memory_size);
    
    if (gpu_vm_get_results(vm, registers, memory) != 0) {
        const char* error = gpu_vm_get_last_error(vm);
        LOG_ERROR("Error: Failed to get results: %s\n", error);
        free(registers);
        free(memory);
        bytecode_destroy(program);
        gpu_vm_destroy(vm);
        return 1;
    }
    
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_COLOR(COLOR_HEADER, "Results (Thread 0):\n");
    LOG_COLOR(COLOR_LABEL, "Registers:\n");
    int regs_shown = 0;
    for (int i = 0; i < 32; i++) {
        if (registers[i] != 0) {
            LOG_COLOR(COLOR_INFO, "  R%d", i);
            LOG_COLOR(COLOR_RESET, " = ");
            LOG_COLOR(COLOR_VALUE, "%d", registers[i]);
            LOG_COLOR(COLOR_METADATA, " (0x%08X)\n", registers[i]);
            regs_shown++;
        }
    }
    if (regs_shown == 0) {
        LOG_COLOR(COLOR_METADATA, "  (all registers are zero)\n");
        }
    
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_COLOR(COLOR_LABEL, "Memory (first 16 words):\n");
    for (int i = 0; i < 16; i++) {
        if (memory[i] != 0) {
            LOG_COLOR(COLOR_INFO, "  [%d]", i);
            LOG_COLOR(COLOR_RESET, " = ");
            LOG_COLOR(COLOR_VALUE, "%d", memory[i]);
            LOG_COLOR(COLOR_METADATA, " (0x%08X)", memory[i]);
            if (program && program->data_size > 0 && i < (int)(program->data_size / sizeof(int32_t))) {
                LOG_COLOR(COLOR_METADATA, " [from data segment]");
            }
            LOG_COLOR(COLOR_RESET, "\n");
        }
    }
    
    int key_addrs[] = {20, 21, 22, 23, 50, 60, 100, 101, 200, 201, 202};
    int found_any = 0;
    for (int i = 0; i < sizeof(key_addrs)/sizeof(key_addrs[0]); i++) {
        int addr = key_addrs[i];
        if (addr < memory_size && memory[addr] != 0) {
            if (!found_any) {
                LOG_COLOR(COLOR_RESET, "\n");
                LOG_COLOR(COLOR_LABEL, "Memory (key locations):\n");
                found_any = 1;
            }
            LOG_COLOR(COLOR_INFO, "  [%d]", addr);
            LOG_COLOR(COLOR_RESET, " = ");
            LOG_COLOR(COLOR_VALUE, "%d", memory[addr]);
            LOG_COLOR(COLOR_METADATA, " (0x%08X)\n", memory[addr]);
        }
    }
    
    free(registers);
    free(memory);
    bytecode_destroy(program);
    gpu_vm_destroy(vm);
    
    LOG_COLOR(COLOR_RESET, "\n");
    LOG_COLOR(COLOR_SUCCESS, "Done.\n");
    return 0;
}
