#include "utils/timer.h"
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

#ifdef _WIN32
static double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
#endif

void timer_start(Timer* timer) {
    timer->start_time = get_time_ms();
    timer->end_time = 0.0;
}

void timer_stop(Timer* timer) {
    timer->end_time = get_time_ms();
}

double timer_elapsed_ms(const Timer* timer) {
    if (timer->end_time == 0.0) {
        return get_time_ms() - timer->start_time;
    }
    return timer->end_time - timer->start_time;
}

double timer_elapsed_us(const Timer* timer) {
    return timer_elapsed_ms(timer) * 1000.0;
}