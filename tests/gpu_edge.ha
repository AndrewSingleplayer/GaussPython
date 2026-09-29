// GPU edge cases that must behave exactly like the CPU (<kernel>_cpu):
// writes into matrices stored in buffers, atomics that must run once, select() evaluating
// both values, integer division by 0 and -1, f16 math and nested arrays.

struct M { m: mat3, v: vec3, w: [3][2]f32, h: hvec2 }

fn weight(x: f32) -> f32 { return x * 0.5; }

@workgroup(64)
kernel edge(ms: *M, cnt: *u32, hits: *f32, iv: *ivec3, uv: *uvec3, out: *f32, weight: f32, n: u32) {
    let i = global_id.x;
    if i >= n { return; }
    // parts of a matrix in memory
    ms[i].m[1].y = 5.0;
    ms[i].m[2][0] += 1.0;
    ms[i].m[0].xz = vec2(ms[i].v.x, 2.0);
    ms[i].m[(i % 3) as i32][(i % 2) as i32] *= 3.0;
    ms[i].w[2][1] = ms[i].w[0][1] * 2.0 + ms[i].w[1][0];
    // the atomic in the place of a compound assignment runs once per thread
    hits[i * 2 + (atomic_add(cnt, 0, 1) & 0)] += 1.0;
    // select evaluates both values (so both atomics run)
    let s = select(ms[i].v.x > 0.0, vec3(1.0, 2.0, 3.0), ms[i].v);
    let c = select(i % 2 == 0, atomic_add(cnt, 1, 1), atomic_add(cnt, 2, 1));
    // integer division never faults: x/0 = 0, x%0 = x, MIN/-1 = MIN, x%-1 = 0
    let k = (i as i32) - 30;
    iv[i] = iv[i] / ivec3(0, -1, k) + iv[i] % ivec3(k, 0, -1);
    uv[i] = uv[i] / uvec3(0, 1, i) + uv[i] % uvec3(i, 0, 5);
    // f16 math (computed through f32 on Vulkan)
    let h = ms[i].h;
    ms[i].h = hvec2(sqrt(abs(h.x)), round(h.y * (4.0 as f16)));
    // a parameter and a function with the same name
    out[i] = s.x + s.y + s.z + (c as f32) * 0.0 + mod(ms[i].v.y, 1.5) + (ms[i].v.z % 2.0) + weight(weight);
}
