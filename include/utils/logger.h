#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <stdarg.h>
#include "utils/colors.h"

#ifdef __cplusplus
extern "C" {
#endif

void logger_init(void);

void logger_shutdown(void);

int logger_is_enabled(void);

void logger_printf(const char* color, const char* format, ...);

#define LOG_INFO(...) logger_printf(COLOR_INFO, __VA_ARGS__)

#define LOG_ERROR(...) logger_printf(COLOR_ERROR, __VA_ARGS__)

#define LOG_COLOR(color, ...) logger_printf(color, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif