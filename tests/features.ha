// Language feature battery. The same file runs natively (x86-64) and under an
// ARM64 emulator; every result is written to `out` and the two runs must agree.

struct Out { p: *u8, n: i64 }

fn put_i(o: *Out, v: i64) {
    let slot = (o.p + o.n) as *i64;
    slot[0] = 0;
    slot[1] = v;
    o.n += 16;
}

fn put_f(o: *Out, v: f64) {
    let tag = (o.p + o.n) as *i64;
    tag[0] = 1;
    let val = (o.p + o.n + 8) as *f64;
    val[0] = v;
    o.n += 16;
}

fn put_v(o: *Out, v: vec4) {
    put_f(o, v.x as f64);
    put_f(o, v.y as f64);
    put_f(o, v.z as f64);
    put_f(o, v.w as f64);
}

struct Inner { a: [3]f32, flag: bool, h: f16 }
struct Outer { id: u16, pos: vec3, inner: Inner, m: mat2 }

fn make_outer(id: u16, x: f32) -> Outer {
    var o: Outer;
    o.id = id;
    o.pos = vec3(x, x * 2.0, x * 3.0);
    o.inner.a[1] = x + 0.5;
    o.inner.flag = x > 1.0;
    o.inner.h = (x * 0.25) as f16;
    o.m = mat2(vec2(1.0, x), vec2(-x, 2.0));
    return o;
}

fn bump(counter: *i32, v: bool) -> bool {
    counter[0] += 1;
    return v;
}

fn fib(n: i32) -> i64 {
    var a: i64 = 0;
    var b: i64 = 1;
    var i = 0;
    while true {
        if i >= n { break; }
        let t = a + b;
        a = b;
        b = t;
        i += 1;
    }
    return a;
}

export fn test_main(inp: *u8, out_bytes: *u8) -> i64 {
    var o = Out(p: out_bytes, n: 0);
    let op = &o;

    // ---- integers: wrapping, division, shifts, bits
    let a8: i8 = 100;
    put_i(op, (a8 + 100) as i64);                 // wraps to -56
    let u8v: u8 = 250;
    put_i(op, (u8v + 10) as i64);                 // wraps to 4
    let big: i32 = 2147483647;
    put_i(op, (big + 1) as i64);
    put_i(op, (-7 / 2) as i64);
    put_i(op, (-7 % 2) as i64);
    let u: u32 = 4000000000;
    put_i(op, (u / 3) as i64);
    put_i(op, (u % 7) as i64);
    let s: i32 = 1;
    put_i(op, (s << 33) as i64);                  // shift amount masked -> 2
    let neg: i32 = -16;
    put_i(op, (neg >> 2) as i64);                 // arithmetic
    put_i(op, ((neg as u32) >> 2) as i64);        // logical
    put_i(op, (0xF0F0 & 0x0FF0 | 0x1 ^ 0x3) as i64);
    put_i(op, popcount(0xFFFF00FF as u32) as i64);
    put_i(op, clz(1 as u32) as i64);
    put_i(op, ctz(0x80 as u32) as i64);
    put_i(op, (~(5 as u8)) as i64);
    put_i(op, fib(80));
    let w64: i64 = 3037000499;
    put_i(op, w64 * w64);

    // ---- conversions (float -> int saturates, like Rust)
    put_i(op, (1.0e10 as i32) as i64);
    put_i(op, (-1.0e10 as i32) as i64);
    let nan = f32_from_bits(0x7fc00000);
    put_i(op, (nan as i32) as i64);
    put_i(op, (300.7 as u8) as i64);
    put_i(op, (-5.0 as u8) as i64);
    let bigu: u64 = 18446744073709551615;
    put_f(op, (bigu as f32) as f64);
    put_f(op, ((1.0 / 3.0) as f16) as f64);
    put_f(op, (65519.0 as f16) as f64);
    put_f(op, (1.0e-6 as f16) as f64);
    put_i(op, f32_bits(-2.5) as i64);
    put_i(op, (true as i32) as i64);

    // ---- float math
    let xs = [0.3, -1.7, 2.5, 10.0];
    for i in 0..4 {
        let x = xs[i];
        put_f(op, exp(x) as f64);
        put_f(op, log(abs(x)) as f64);
        put_f(op, sin(x) as f64);
        put_f(op, cos(x) as f64);
        put_f(op, tan(x * 0.1) as f64);
        put_f(op, tanh(x) as f64);
        put_f(op, atan2(x, 1.5) as f64);
        put_f(op, pow(abs(x), 1.7) as f64);
        put_f(op, sqrt(abs(x)) as f64);
        put_f(op, floor(x) as f64);
        put_f(op, ceil(x) as f64);
        put_f(op, round(x) as f64);
        put_f(op, fract(x) as f64);
        put_f(op, mod(x, 0.7) as f64);
        put_f(op, (x % 0.7) as f64);
        put_f(op, sigmoid(x) as f64);
        put_f(op, gelu(x) as f64);
    }
    put_f(op, clamp(5.0, 0.0, 1.0) as f64);
    put_f(op, mix(2.0, 4.0, 0.25) as f64);
    put_f(op, smoothstep(0.0, 2.0, 0.5) as f64);
    put_f(op, step(1.0, 0.5) as f64);
    let d: f64 = 1.0 / 3.0;
    put_f(op, d * 3.0 - 1.0);
    put_f(op, sqrt(2.0 as f64));

    // ---- vectors and matrices
    let v = vec3(1.0, -2.0, 0.5);
    let w = vec3(0.25, 4.0, -1.0);
    put_v(op, vec4(cross(v, w), dot(v, w)));
    put_v(op, vec4(normalize(v), length(w)));
    var sw = vec4(1.0, 2.0, 3.0, 4.0);
    sw.xz = sw.zx;
    sw.w = sw.y * 10.0;
    sw[1] = -sw[1];
    put_v(op, sw);
    put_v(op, vec4(v.zyx * 2.0 + w.xxy, max(v, w).y));
    let iv = ivec3(7, -3, 2);
    let iv2 = iv * 3 + ivec3(1) >> ivec3(1, 0, 2);
    put_i(op, iv2.x as i64);
    put_i(op, iv2.y as i64);
    put_i(op, iv2.z as i64);
    let hv = hvec4(1.5 as f16, 2.0 as f16, 0.1 as f16, 3.0 as f16) * (2.0 as f16);
    put_v(op, vec4(hv.x as f32, hv.y as f32, hv.z as f32, hv.w as f32));
    let m = mat4(vec4(1.0, 2.0, 3.0, 4.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.5, 0.0, 1.0, 0.0),
                 vec4(0.0, 0.0, 0.0, 1.0));
    put_v(op, m * vec4(1.0, 2.0, 3.0, 1.0));
    put_v(op, (transpose(m) * m) * vec4(0.5, -1.0, 2.0, 1.0));
    let q = quat_to_mat3(vec4(0.9238795, 0.0, 0.3826834, 0.0));
    put_v(op, vec4(q * vec3(1.0, 0.0, 0.0), q[1].y));
    put_i(op, pack_unorm4(vec4(1.0, 0.5, 0.0, 0.2)) as i64);
    put_v(op, unpack_unorm4(0x80FF4000));
    put_i(op, pack_half2(vec2(1.0, -2.5)) as i64);
    let uh = unpack_half2(0x3C00C100);
    put_f(op, uh.x as f64);
    put_f(op, uh.y as f64);

    // ---- structs, arrays, pointers
    var ob = make_outer(9, 1.5);
    ob.inner.a[2] = ob.inner.a[1] * 2.0;
    let copy = ob;
    ob.pos.y = 100.0;
    put_i(op, copy.id as i64);
    put_v(op, vec4(copy.pos, ob.pos.y));
    put_v(op, vec4(copy.inner.a[0], copy.inner.a[1], copy.inner.a[2], copy.inner.h as f32));
    put_i(op, copy.inner.flag as i64);
    put_v(op, vec4(copy.m * vec2(1.0, 1.0), copy.m[1]));
    var grid: [4][3]i32;
    for r in 0..4 {
        for c in 0..3 {
            if c == 1 { continue; }
            grid[r][c] = r * 10 + c;
        }
    }
    put_i(op, (grid[3][2] + grid[2][0] + grid[1][1]) as i64);
    var arr = [1.0, 2.0, 3.0, 4.0, 5.0];
    let pa = &arr[1];
    pa[0] = 20.0;
    (pa + 2)[0] += 0.5;
    var total = 0.0;
    for i in 0..5 { total += arr[i]; }
    put_f(op, total as f64);

    // ---- control flow and short-circuit evaluation
    var calls: i32 = 0;
    let r1 = bump(&calls, false) && bump(&calls, true);
    let r2 = bump(&calls, true) || bump(&calls, true);
    put_i(op, calls as i64);
    put_i(op, (r1 as i32 + r2 as i32 * 2) as i64);
    put_i(op, select(calls > 2, 111, 222) as i64);

    // ---- read input: n floats, reduce them (vectorized loop)
    let n = (inp as *i64)[0];
    let data = (inp + 8) as *f32;
    var acc = 0.0;
    var acc_s = 0.0;
    for i in 0..n {
        acc += data[i] * data[i];
        acc_s += sigmoid(data[i]);
    }
    put_f(op, acc as f64);
    put_f(op, acc_s as f64);
    return o.n;
}
