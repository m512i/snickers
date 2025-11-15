#include "utils/colors.h"
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif

static int colors_enabled = 0;

int colors_enable(void) {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) {
        return 0;
    }
    
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) {
        return 0;
    }
    
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(hOut, dwMode)) {
        return 0;
    }
    
    colors_enabled = 1;
    return 1;
#else
    if (isatty(fileno(stdout))) {
        colors_enabled = 1;
        return 1;
    }
    return 0;
#endif
}

int colors_disable(void) {
    colors_enabled = 0;
    return 1;
}

int colors_is_enabled(void) {
    return colors_enabled;
}