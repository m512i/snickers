#include "utils/logger.h"
#include <stdio.h>
#include <stdarg.h>

void logger_init(void) {
    colors_enable();
}

void logger_shutdown(void) {
    colors_disable();
}

int logger_is_enabled(void) {
    return colors_is_enabled();
}

void logger_printf(const char* color, const char* format, ...) {
    if (!colors_is_enabled()) {
        colors_enable();
    }
    
    va_list args;
    va_start(args, format);
    
    if (colors_is_enabled() && color) {
        printf("%s", color);
    }
    
    vprintf(format, args);
    
    if (colors_is_enabled()) {
        printf("%s", COLOR_RESET);
    }
    
    va_end(args);
}