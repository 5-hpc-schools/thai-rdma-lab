/*
 * timer.h — เครื่องมือวัดเวลาสำหรับทดสอบประสิทธิภาพ
 */
#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <stdint.h>

/* คืนค่าเวลาปัจจุบันเป็นไมโครวินาที */
static inline uint64_t get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/* คำนวณเวลาเฉลี่ยจากอาร์เรย์ของเวลาแฝง */
static inline double avg_latency_us(uint64_t *latencies, int count)
{
    uint64_t sum = 0;
    for (int i = 0; i < count; i++) {
        sum += latencies[i];
    }
    return (double)sum / count;
}

/* พิมพ์ผลเปรียบเทียบประสิทธิภาพ */
static inline void print_benchmark(const char *name, int msg_size,
                                    double latency_us, int iterations)
{
    double bw_mbps = 0;
    if (latency_us > 0 && msg_size > 0) {
        bw_mbps = (double)msg_size / latency_us; /* เมกะไบต์ต่อวินาที */
    }
    printf("%-25s  ขนาด: %8d ไบต์  เวลาแฝง: %8.2f ไมโครวินาที"
           "  แบนด์วิดท์: %8.2f เมกะไบต์/วินาที  (%d รอบ)\n",
           name, msg_size, latency_us, bw_mbps, iterations);
}

#endif /* TIMER_H */
