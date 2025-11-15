#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "vm/bytecode.h"
#include "vm/vm_state.h"
#include "gpu/gpu_vm.h"
#include "utils/dump.h"
#include "utils/timer.h"
#include "utils/colors.h"

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
    colors_enable();
    
    printf(COLOR_TITLE "GPU Bytecode VM" COLOR_RESET "\n");
    printf(COLOR_TITLE "===============" COLOR_RESET "\n\n");
    
    size_t num_threads = 1024;
    size_t memory_size = 65536;  
    int max_iterations = 10000;
    const char* program_file = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            memory_size = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            max_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            program_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            printf(COLOR_TITLE "Usage: %s [options]" COLOR_RESET "\n", argv[0]);
            printf(COLOR_LABEL "Options:" COLOR_RESET "\n");
            printf("  " COLOR_INFO "-t <num>" COLOR_RESET "    Number of threads (default: 1024)\n");
            printf("  " COLOR_INFO "-m <size>" COLOR_RESET "   Memory size in bytes (default: 65536)\n");
            printf("  " COLOR_INFO "-i <num>" COLOR_RESET "    Max iterations per thread (default: 10000)\n");
            printf("  " COLOR_INFO "-f <file>" COLOR_RESET "   Bytecode program file\n");
            printf("  " COLOR_INFO "-h" COLOR_RESET "          Show this help\n");
            return 0;
        }
    }
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf(COLOR_ERROR "Error: No CUDA devices found or CUDA not available." COLOR_RESET "\n");
        return 1;
    }
    
    printf(COLOR_INFO "Found " COLOR_VALUE "%d" COLOR_INFO " CUDA device(s)" COLOR_RESET "\n", device_count);
    
    GPUVirtualMachine* vm = gpu_vm_create(memory_size, num_threads, 0);
    if (!vm) {
        printf(COLOR_ERROR "Error: Failed to create GPU VM" COLOR_RESET "\n");
        return 1;
    }
    
    gpu_vm_print_info(vm);
    printf("\n");
    
    BytecodeProgram* program = NULL;
    if (program_file) {
        printf(COLOR_INFO "Loading program from file: " COLOR_VALUE "%s" COLOR_RESET "\n", program_file);
        BytecodeProgram temp_program = {0};
        int result = bytecode_load_from_file(program_file, &temp_program);
        if (result != 0) {
            printf(COLOR_ERROR "Error: Failed to load program from file: %s (error code: %d)" COLOR_RESET "\n", program_file, result);
            gpu_vm_destroy(vm);
            return 1;
        }
        program = bytecode_create(temp_program.instruction_count, temp_program.data_size);
        if (!program) {
            printf(COLOR_ERROR "Error: Failed to allocate program memory" COLOR_RESET "\n");
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
    } else {
        printf(COLOR_INFO "Creating sample program: " COLOR_HIGHLIGHT_TEXT "ADD test" COLOR_RESET "\n");
        program = bytecode_create(10, 0);
        if (program) {
            program->instructions[0] = make_instruction(OP_LOAD_IMM, (0 << 16) | 5);
            program->instructions[1] = make_instruction(OP_LOAD_IMM, (1 << 16) | 3);
            program->instructions[2] = make_instruction(OP_ADD, (0 << 16) | (0 << 8) | 1);
            program->instructions[3] = make_instruction(OP_STORE, (0 << 16) | 0);
            program->instructions[4] = make_instruction(OP_HALT, 0);
            
            for (int i = 5; i < 10; i++) {
                program->instructions[i] = make_instruction(OP_NOP, 0);
            }
        }
    }
    
    if (!program) {
        printf(COLOR_ERROR "Error: Failed to create program" COLOR_RESET "\n");
        gpu_vm_destroy(vm);
        return 1;
    }
    
    printf("\n" COLOR_HEADER "Program:" COLOR_RESET "\n");
    dump_program(program);
    
    if (gpu_vm_load_program(vm, program) != 0) {
        const char* error = gpu_vm_get_last_error(vm);
        printf(COLOR_ERROR "Error: Failed to load program: %s" COLOR_RESET "\n", error);
        bytecode_destroy(program);
        gpu_vm_destroy(vm);
        return 1;
    }
    
    printf("\n" COLOR_HEADER "GPU VM Info (after loading program):" COLOR_RESET "\n");
    gpu_vm_print_info(vm);
    printf("\n");
    
    printf("\n" COLOR_INFO "Executing on GPU..." COLOR_RESET "\n");
    Timer timer;
    timer_start(&timer);
    
    if (gpu_vm_execute(vm, max_iterations) != 0) {
        const char* error = gpu_vm_get_last_error(vm);
        printf(COLOR_ERROR "Error: GPU execution failed: %s" COLOR_RESET "\n", error);
        bytecode_destroy(program);
        gpu_vm_destroy(vm);
        return 1;
    }
    
    timer_stop(&timer);
    
    float kernel_time = 0.0f, transfer_time = 0.0f;
    gpu_vm_get_performance_stats(vm, &kernel_time, &transfer_time);
    
    printf(COLOR_LABEL "Execution time: " COLOR_VALUE "%.3f ms" COLOR_METADATA " (Host timer)" COLOR_RESET "\n", timer_elapsed_ms(&timer));
    if (kernel_time > 0) {
        printf(COLOR_LABEL "Kernel execution: " COLOR_VALUE "%.3f ms" COLOR_METADATA " (CUDA event)" COLOR_RESET "\n", kernel_time);
    }
    if (transfer_time > 0) {
        printf(COLOR_LABEL "Memory transfer: " COLOR_VALUE "%.3f ms" COLOR_METADATA " (CUDA event)" COLOR_RESET "\n", transfer_time);
    }
    
    int32_t* registers = (int32_t*)malloc(sizeof(int32_t) * num_threads * 32);
    int32_t* memory = (int32_t*)malloc(sizeof(int32_t) * memory_size);
    
    if (gpu_vm_get_results(vm, registers, memory) != 0) {
        const char* error = gpu_vm_get_last_error(vm);
        printf(COLOR_ERROR "Error: Failed to get results: %s" COLOR_RESET "\n", error);
        free(registers);
        free(memory);
        bytecode_destroy(program);
        gpu_vm_destroy(vm);
        return 1;
    }
    
    printf("\n" COLOR_HEADER "Results (Thread 0):" COLOR_RESET "\n");
    printf(COLOR_LABEL "Registers:" COLOR_RESET "\n");
    int regs_shown = 0;
    for (int i = 0; i < 32; i++) {
        if (registers[i] != 0) {
            printf("  " COLOR_INFO "R%d" COLOR_RESET " = " COLOR_VALUE "%d" COLOR_METADATA " (0x%08X)" COLOR_RESET "\n", i, registers[i], registers[i]);
            regs_shown++;
        }
    }
    if (regs_shown == 0) {
        printf("  " COLOR_METADATA "(all registers are zero)" COLOR_RESET "\n");
    }
    
    printf("\n" COLOR_LABEL "Memory (first 16 words):" COLOR_RESET "\n");
    for (int i = 0; i < 16; i++) {
        if (memory[i] != 0) {
            printf("  " COLOR_INFO "[%d]" COLOR_RESET " = " COLOR_VALUE "%d" COLOR_METADATA " (0x%08X)" COLOR_RESET "\n", i, memory[i], memory[i]);
        }
    }
    
    int key_addrs[] = {50, 60, 100, 101, 200, 201, 202};
    int found_any = 0;
    for (int i = 0; i < sizeof(key_addrs)/sizeof(key_addrs[0]); i++) {
        int addr = key_addrs[i];
        if (addr < memory_size && memory[addr] != 0) {
            if (!found_any) {
                printf("\n" COLOR_LABEL "Memory (key locations):" COLOR_RESET "\n");
                found_any = 1;
            }
            printf("  " COLOR_INFO "[%d]" COLOR_RESET " = " COLOR_VALUE "%d" COLOR_METADATA " (0x%08X)" COLOR_RESET "\n", addr, memory[addr], memory[addr]);
        }
    }
    
    free(registers);
    free(memory);
    bytecode_destroy(program);
    gpu_vm_destroy(vm);
    
    printf("\n" COLOR_SUCCESS "Done." COLOR_RESET "\n");
    return 0;
}
