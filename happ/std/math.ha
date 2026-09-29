// HA++ standard library.
//
// On CPUs the math builtins (exp, log, sin, ...) are implemented here, in
// HA++ itself, so compiled libraries need no C math library. Every function is
// branch-free (select instead of if), which lets LLVM inline and vectorize
// them inside loops (4-8 lanes at a time with NEON/AVX).
// On GPUs the builtins map to the GPU's own instructions instead.
//
// Accuracy (f32): exp/exp2/log/log2/sin/cos/atan/asin/tanh stay within a few
// ulp; sin/cos/tan are accurate for |x| < 8192; pow = exp(y*log(x)).
// The algorithms follow the Cephes single-precision library.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const E: f32 = 2.71828182845905;

const _INF_BITS: u32 = 0x7f800000;
const _NEG_INF_BITS: u32 = 0xff800000;
const _NAN_BITS: u32 = 0x7fc00000;

// ---------------------------------------------------------------- public helpers

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn relu(x: f32) -> f32 { return max(x, 0.0); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn softplus(x: f32) -> f32 { return select(x > 20.0, x, log(1.0 + exp(x))); }
fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn radians(deg: f32) -> f32 { return deg * 0.0174532925199433; }
fn degrees(rad: f32) -> f32 { return rad * 57.2957795130823; }

// GELU, tanh approximation (as used by GPT-style models)
fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

// Quaternion to rotation matrix. Pass it as vec4(w, x, y, z): the order
// Gaussian splat .ply files use. It is normalized first.
fn quat_to_mat3(q: vec4) -> mat3 {
    let n = normalize(q);
    let w = n.x;
    let x = n.y;
    let y = n.z;
    let z = n.w;
    return mat3(
        vec3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

// ---------------------------------------------------------------- internals

// y * 2^k for k in [-252, 254], using two exact power-of-two scalings so that
// results that overflow or become subnormal are rounded only once.
@strict
fn _std_ldexp(y: f32, k: i32) -> f32 {
    let k1 = k >> 1;
    let k2 = k - k1;
    let s1 = f32_from_bits(((k1 + 127) as u32) << 23);
    let s2 = f32_from_bits(((k2 + 127) as u32) << 23);
    return (y * s1) * s2;
}

// e^r for |r| <= ln(2)/2
fn _std_exp_poly(r: f32) -> f32 {
    let z = r * r;
    return (((((1.9875691500e-4 * r + 1.3981999507e-3) * r + 8.3334519073e-3) * r
             + 4.1665795894e-2) * r + 1.6666665459e-1) * r + 5.0000001201e-1) * z + r + 1.0;
}

fn _std_exp(x: f32) -> f32 {
    let xc = clamp(x, -104.0, 89.0);
    let k = floor(xc * 1.44269504088896341 + 0.5);
    let r = (xc - k * 0.693359375) - k * -2.12194440e-4;
    let y = _std_ldexp(_std_exp_poly(r), k as i32);
    return select(x != x, x, y);
}

fn _std_exp2(x: f32) -> f32 {
    let xc = clamp(x, -151.0, 129.0);
    let k = floor(xc + 0.5);
    let r = (xc - k) * 0.693147180559945309;
    let y = _std_ldexp(_std_exp_poly(r), k as i32);
    return select(x != x, x, y);
}

// log(x) = fe*ln2 + log(1+m): returns vec3(m, t, fe) with log(1+m) ~= m + t
fn _std_log_parts(x: f32) -> vec3 {
    let sub = f32_bits(x) < 0x00800000;
    let xs = select(sub, x * 8388608.0, x);
    let bits = f32_bits(xs);
    var e = ((bits >> 23) as i32) - 126 - select(sub, 23, 0);
    var m = f32_from_bits((bits & 0x007fffff) | 0x3f000000);
    let small = m < 0.707106781186547524;
    e = e - select(small, 1, 0);
    m = select(small, m + m - 1.0, m - 1.0);
    let z = m * m;
    let p = ((((((((7.0376836292e-2 * m - 1.1514610310e-1) * m + 1.1676998740e-1) * m
            - 1.2420140846e-1) * m + 1.4249322787e-1) * m - 1.6668057665e-1) * m
            + 2.0000714765e-1) * m - 2.4999993993e-1) * m + 3.3333331174e-1) * m * z;
    return vec3(m, p - 0.5 * z, e as f32);
}

fn _std_log_special(x: f32, r: f32) -> f32 {
    var y = select(x == 0.0, f32_from_bits(_NEG_INF_BITS), r);
    y = select(x < 0.0, f32_from_bits(_NAN_BITS), y);
    y = select(x == f32_from_bits(_INF_BITS), x, y);
    return select(x != x, x, y);
}

fn _std_log(x: f32) -> f32 {
    let p = _std_log_parts(x);
    let y = p.y + p.z * -2.12194440e-4;
    let r = (p.x + y) + p.z * 0.693359375;
    return _std_log_special(x, r);
}

fn _std_log2(x: f32) -> f32 {
    let p = _std_log_parts(x);
    let r = (p.x + p.y) * 1.44269504088896341 + p.z;
    return _std_log_special(x, r);
}

// Cody-Waite reduction by pi/4; returns vec2(reduced x, octant)
fn _std_trig_reduce(ax: f32) -> vec2 {
    var j = (ax * 1.27323954473516) as i32;
    j = j + (j & 1);
    let y = j as f32;
    let r = ((ax - y * 0.78515625) - y * 2.4187564849853515625e-4) - y * 3.77489497744594108e-8;
    return vec2(r, (j & 7) as f32);
}

fn _std_sin_poly(x: f32, z: f32) -> f32 {
    return ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1) * z * x + x;
}

fn _std_cos_poly(z: f32) -> f32 {
    return ((2.443315711809948e-5 * z - 1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z
        - 0.5 * z + 1.0;
}

fn _std_sin(x: f32) -> f32 {
    let ax = abs(x);
    let red = _std_trig_reduce(ax);
    var j = red.y as i32;
    var neg = x < 0.0;
    neg = neg != (j > 3);
    j = select(j > 3, j - 4, j);
    let z = red.x * red.x;
    let use_cos = j == 1 || j == 2;
    let y = select(use_cos, _std_cos_poly(z), _std_sin_poly(red.x, z));
    let r = select(neg, -y, y);
    return select(ax == f32_from_bits(_INF_BITS), f32_from_bits(_NAN_BITS), r);
}

fn _std_cos(x: f32) -> f32 {
    let ax = abs(x);
    let red = _std_trig_reduce(ax);
    var j = red.y as i32;
    var neg = j > 3;
    j = select(j > 3, j - 4, j);
    neg = neg != (j > 1);
    let z = red.x * red.x;
    let use_sin = j == 1 || j == 2;
    let y = select(use_sin, _std_sin_poly(red.x, z), _std_cos_poly(z));
    let r = select(neg, -y, y);
    return select(ax == f32_from_bits(_INF_BITS), f32_from_bits(_NAN_BITS), r);
}

fn _std_tan(x: f32) -> f32 {
    return _std_sin(x) / _std_cos(x);
}

fn _std_atan(x: f32) -> f32 {
    let ax = abs(x);
    let big = ax > 2.414213562373095;
    let mid = ax > 0.4142135623730950;
    let xr = select(big, -1.0 / ax, select(mid, (ax - 1.0) / (ax + 1.0), ax));
    let base = select(big, 1.5707963267948966, select(mid, 0.7853981633974483, 0.0));
    let z = xr * xr;
    let y = base + ((((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z
                     - 3.33329491539e-1) * z * xr + xr);
    return select(x < 0.0, -y, y);
}

fn _std_atan2(y: f32, x: f32) -> f32 {
    let a = _std_atan(y / x);
    var r = select(x < 0.0, a + select(y < 0.0, -3.14159265358979, 3.14159265358979), a);
    r = select(x == 0.0, select(y > 0.0, 1.5707963267948966, select(y < 0.0, -1.5707963267948966, 0.0)), r);
    return r;
}

fn _std_asin(x: f32) -> f32 {
    let a = abs(x);
    let flag = a > 0.5;
    let z = select(flag, 0.5 * (1.0 - a), a * a);
    let xx = select(flag, sqrt(z), a);
    var y = ((((4.2163199048e-2 * z + 2.4181311049e-2) * z + 4.5470025998e-2) * z
              + 7.4953002686e-2) * z + 1.6666752422e-1) * z * xx + xx;
    y = select(flag, 1.5707963267948966 - (y + y), y);
    y = select(x < 0.0, -y, y);
    return select(a > 1.0, f32_from_bits(_NAN_BITS), y);
}

fn _std_acos(x: f32) -> f32 {
    let lo = x < -0.5;
    let hi = x > 0.5;
    let arg = select(lo, sqrt(0.5 * (1.0 + x)), select(hi, sqrt(0.5 * (1.0 - x)), x));
    let a = _std_asin(arg);
    return select(lo, 3.14159265358979 - 2.0 * a, select(hi, 2.0 * a, 1.5707963267948966 - a));
}

fn _std_tanh(x: f32) -> f32 {
    let z = abs(x);
    let s = _std_exp(2.0 * min(z, 44.0));
    let big = 1.0 - 2.0 / (s + 1.0);
    let x2 = x * x;
    let small = ((((-5.70498872745e-3 * x2 + 2.06390887954e-2) * x2 - 5.37397155531e-2) * x2
                  + 1.33314422036e-1) * x2 - 3.33332819422e-1) * x2 * x + x;
    let r = select(z < 0.625, small, select(x < 0.0, -big, big));
    return select(x != x, x, r);
}

fn _std_pow(x: f32, y: f32) -> f32 {
    let ax = abs(x);
    var r = _std_exp(y * _std_log(ax));
    // negative base: defined for whole-number exponents
    let whole = floor(y) == y;
    let odd = whole && floor(y * 0.5) != y * 0.5;
    r = select(x < 0.0, select(whole, select(odd, -r, r), f32_from_bits(_NAN_BITS)), r);
    r = select(x == 0.0, select(y > 0.0, 0.0, select(y < 0.0, f32_from_bits(_INF_BITS), 1.0)), r);
    r = select(y == 0.0 || x == 1.0, 1.0, r);
    return r;
}

// ---------------------------------------------------------------- color packing
// RGBA in [0,1] <-> one u32 (8 bits per channel, red in the low byte).
// Plain HA++ code, so CPU and GPU produce identical bits.

fn pack_unorm4(c: vec4) -> u32 {
    let u = (clamp(c, 0.0, 1.0) * 255.0 + 0.5) as uvec4;
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

fn unpack_unorm4(u: u32) -> vec4 {
    return vec4((u & 255) as f32, ((u >> 8) & 255) as f32, ((u >> 16) & 255) as f32, (u >> 24) as f32)
        * 0.00392156862745098;
}
