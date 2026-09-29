/* Freestanding Linux/AArch64 runner (no C library): reads stdin, calls the HA++
   function test_main(in, out), writes the result bytes to stdout. Used with qemu-aarch64. */
long long test_main(unsigned char *in, unsigned char *out);

static long sys3(long n, long a, long b, long c) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a;
    register long x1 __asm__("x1") = b;
    register long x2 __asm__("x2") = c;
    __asm__ volatile("svc 0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2) : "memory");
    return x0;
}

static unsigned char in_buf[1 << 22] __attribute__((aligned(16)));
static unsigned char out_buf[1 << 20] __attribute__((aligned(16)));

void _start(void) {
    long n = 0, r;
    while ((r = sys3(63 /* read */, 0, (long)(in_buf + n), (long)sizeof(in_buf) - n)) > 0) n += r;
    long long m = test_main(in_buf, out_buf);
    long w = 0;
    while (w < m && (r = sys3(64 /* write */, 1, (long)(out_buf + w), m - w)) > 0) w += r;
    sys3(93 /* exit */, 0, 0, 0);
}
