#ifndef COLORS_H
#define COLORS_H

#ifdef __cplusplus
extern "C" {
#endif

int colors_enable(void);
int colors_disable(void);
int colors_is_enabled(void);

#define COLOR_RESET       "\033[0m"

#define COLOR_BLACK       "\033[30m"
#define COLOR_RED         "\033[31m"
#define COLOR_GREEN       "\033[32m"
#define COLOR_YELLOW      "\033[33m"
#define COLOR_BLUE        "\033[34m"
#define COLOR_MAGENTA     "\033[35m"
#define COLOR_CYAN        "\033[36m"
#define COLOR_WHITE       "\033[37m"

#define COLOR_BRIGHT_BLACK   "\033[90m"
#define COLOR_BRIGHT_RED     "\033[91m"
#define COLOR_BRIGHT_GREEN   "\033[92m"
#define COLOR_BRIGHT_YELLOW  "\033[93m"
#define COLOR_BRIGHT_BLUE    "\033[94m"
#define COLOR_BRIGHT_MAGENTA "\033[95m"
#define COLOR_BRIGHT_CYAN    "\033[96m"
#define COLOR_BRIGHT_WHITE   "\033[97m"

#define COLOR_BOLD        "\033[1m"
#define COLOR_DIM         "\033[2m"
#define COLOR_UNDERLINE   "\033[4m"

#define COLOR_HEADER      COLOR_BRIGHT_CYAN COLOR_BOLD
#define COLOR_TITLE       COLOR_BRIGHT_BLUE COLOR_BOLD
#define COLOR_SUCCESS     COLOR_BRIGHT_GREEN
#define COLOR_INFO        COLOR_CYAN
#define COLOR_WARNING     COLOR_YELLOW
#define COLOR_ERROR       COLOR_BRIGHT_RED
#define COLOR_VALUE       COLOR_GREEN
#define COLOR_LABEL       COLOR_BRIGHT_WHITE
#define COLOR_METADATA    COLOR_BRIGHT_BLACK
#define COLOR_HIGHLIGHT_TEXT COLOR_BRIGHT_YELLOW

#ifdef __cplusplus
}
#endif

#endif