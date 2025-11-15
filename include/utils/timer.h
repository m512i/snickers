#ifndef TIMER_H
#define TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double start_time;
    double end_time;
} Timer;

void timer_start(Timer* timer);
void timer_stop(Timer* timer);
double timer_elapsed_ms(const Timer* timer);
double timer_elapsed_us(const Timer* timer);

#ifdef __cplusplus
}
#endif

#endif