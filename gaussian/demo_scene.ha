// A procedural "galaxy" of Gaussian splats, so the apps have something to show
// without shipping a .ply file. Plain CPU code; same result on every device.

import "splat.ha";

// PCG-style integer hash -> pseudo random numbers without any state
fn hash32(x: u32) -> u32 {
    let s = x * 747796405 + 2891336453;
    let w = ((s >> ((s >> 28) + 4)) ^ s) * 277803737;
    return (w >> 22) ^ w;
}

fn rand01(k: u32) -> f32 {
    return (hash32(k) >> 8) as f32 * 0.000000059604644775390625;     // [0, 1)
}

fn gauss(k: u32) -> f32 {
    let u1 = max(rand01(k), 0.0000001);
    let u2 = rand01(k ^ 0x9E3779B9);
    return sqrt(-2.0 * log(u1)) * cos(TAU * u2);
}

export fn make_galaxy(out: *GaussianRaw, n: i64, seed: u32) {
    for i in 0..n {
        let k = (i as u32) * 16 + seed * 7919;
        let arm = (hash32(k) % 3) as f32;
        let r = (-log(max(rand01(k + 1), 0.0000001)) - log(max(rand01(k + 2), 0.0000001))) * 0.6;
        let theta = r * 1.8 + arm * (TAU / 3.0) + gauss(k + 3) * 0.25;
        var g: GaussianRaw;
        g.pos = vec3(r * cos(theta), gauss(k + 4) * 0.08 * (1.0 + 0.5 / (r + 0.3)), r * sin(theta));
        g.log_scale = log(vec3(0.02) + vec3(rand01(k + 5), rand01(k + 6), rand01(k + 7)) * 0.07);
        g.rot = normalize(vec4(gauss(k + 8), gauss(k + 9), gauss(k + 10), gauss(k + 11)));
        g.opacity_logit = 1.5 + gauss(k + 12);
        let hue = fract(theta / TAU + r * 0.15);
        let rgb = (0.5 + 0.5 * cos(TAU * (vec3(hue) + vec3(0.0, 0.333333333, 0.666666667)))) * 0.8
            + vec3(0.2 * exp(-r));
        g.sh_dc = (rgb - 0.5) / SH_C0;
        out[i] = g;
    }
}
