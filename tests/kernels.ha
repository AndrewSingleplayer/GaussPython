// GPU test kernels: struct buffers + matrices (project), shared memory + atomics (hist),
// f16 buffers (halfs) and subgroup operations (scan). Used by tests/test_happ.py.

struct Splat { pos: vec3, scale: vec3, rot: vec4, opacity: f32, sh: [3]f32 }
struct Out2D { center: vec2, conic: vec3, radius: f32, depth: f32 }

fn cov2d(s: Splat, view: mat3) -> vec3 {
    let r = quat_to_mat3(s.rot);
    let m = r * mat3(vec3(s.scale.x, 0.0, 0.0), vec3(0.0, s.scale.y, 0.0), vec3(0.0, 0.0, s.scale.z));
    let c = view * (m * transpose(m)) * transpose(view);
    return vec3(c[0].x, c[0].y, c[1].y);
}

@workgroup(64)
kernel project(splats: *Splat, out: *Out2D, n: u32, focal: f32) {
    let i = global_id.x;
    if i >= n { return; }
    let s = splats[i];
    var o: Out2D;
    let c = cov2d(s, mat3(1.0));
    o.conic = c * focal;
    o.center = s.pos.xy / max(s.pos.z, 0.01) * focal;
    o.radius = 3.0 * sqrt(max(c.x, c.z));
    o.depth = s.pos.z;
    o.center.y = -o.center.y;
    out[i] = o;
    out[i].conic.x = sigmoid(s.opacity);
    var sh = s.sh;
    sh[1] += 1.0;
    out[i].radius += sh[1] + (i % 7) as f32 + f32_from_bits(0x3f800000) + fract(1.25) + round(2.5);
}

@workgroup(256)
kernel hist(keys: *u32, counts: *u32, n: u32) {
    shared local_counts: [256]u32;
    let t = local_id.x;
    local_counts[t] = 0;
    barrier();
    let i = global_id.x;
    if i < n {
        let b = (keys[i] >> 24) & 255;
        atomic_add(local_counts, b, 1);
    }
    barrier();
    atomic_add(counts, t, local_counts[t]);
}

@workgroup(64)
kernel halfs(a: *f16, b: *hvec4, n: u32) {
    let i = global_id.x;
    if i < n { b[i] = hvec4(a[i]) * (2.0 as f16); b[i].w = a[i]; }
}

@workgroup(32)
kernel scan(a: *u32) {
    let i = global_id.x;
    a[i] = subgroup_exclusive_add(a[i]) + subgroup_size() * 0;
}
