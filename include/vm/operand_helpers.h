#ifndef OPERAND_HELPERS_H
#define OPERAND_HELPERS_H

#include <stdint.h>

#define EXTRACT_REG_DST(op) ((op) >> 16) & 0xFF
#define EXTRACT_REG_SRC1(op) ((op) >> 8) & 0xFF
#define EXTRACT_REG_SRC2(op) (op) & 0xFF
#define EXTRACT_IMMEDIATE(op) (int32_t)((op) & 0xFFFF)
#define EXTRACT_ADDRESS(op) (op) & 0xFFFF
#define EXTRACT_JUMP_TARGET(op) (op) & 0xFFFF

#define IS_VALID_REG(reg_idx, max_regs) ((reg_idx) < (max_regs))
#define IS_VALID_REG_PAIR(reg1, reg2, max_regs) \
    (IS_VALID_REG(reg1, max_regs) && IS_VALID_REG(reg2, max_regs))
#define IS_VALID_REG_TRIPLE(reg1, reg2, reg3, max_regs) \
    (IS_VALID_REG(reg1, max_regs) && IS_VALID_REG(reg2, max_regs) && IS_VALID_REG(reg3, max_regs))

#endif