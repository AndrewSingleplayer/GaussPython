/* HA++ host runtime: only used by `happ run` (programs with a main).
   Libraries built for phones never link this file. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

void ha_rt_print_i64(int64_t v) { printf("%" PRId64, v); }
void ha_rt_print_u64(uint64_t v) { printf("%" PRIu64, v); }
void ha_rt_print_f32(float v) { printf("%.7g", (double)v); }
void ha_rt_print_f64(double v) { printf("%.15g", v); }
void ha_rt_print_bool(int32_t v) { fputs(v ? "true" : "false", stdout); }
void ha_rt_print_ptr(void *p) { printf("%p", p); }
void ha_rt_print_str(const char *s, int64_t n) { fwrite(s, 1, (size_t)n, stdout); }
void ha_rt_print_space(void) { putchar(' '); }
void ha_rt_print_open(void) { putchar('('); }
void ha_rt_print_comma(void) { fputs(", ", stdout); }
void ha_rt_print_close(void) { putchar(')'); }
void ha_rt_print_end(void) { putchar('\n'); }
