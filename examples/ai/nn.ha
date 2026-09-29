// Neural-network building blocks in HA++.
//
// Every operation exists twice: as a GPU kernel (Vulkan on Android/PC, Metal
// on iPhone/Mac) and as a CPU function (export fn) that LLVM vectorizes for
// NEON / AVX2. Matrices are row-major f32.
// Together they run a transformer-style MLP block:  softmax(gelu(x W1 + b1) W2)

// ---------------------------------------------------------------- GPU kernels

// C[m x n] = A[m x k] * B[k x n], 16x16 tiles staged in shared memory.
@workgroup(16, 16)
kernel matmul(a: *f32, b: *f32, c: *f32, m: u32, n: u32, k: u32) {
    shared ta: [256]f32;
    shared tb: [256]f32;
    let row = global_id.y;
    let col = global_id.x;
    let tx = local_id.x;
    let ty = local_id.y;
    var acc = 0.0;
    var t0: u32 = 0;
    while t0 < k {
        var av = 0.0;
        if row < m && t0 + tx < k { av = a[row * k + t0 + tx]; }
        var bv = 0.0;
        if col < n && t0 + ty < k { bv = b[(t0 + ty) * n + col]; }
        ta[ty * 16 + tx] = av;
        tb[ty * 16 + tx] = bv;
        barrier();
        for j: u32 in 0..16 { acc += ta[ty * 16 + j] * tb[j * 16 + tx]; }
        barrier();
        t0 += 16;
    }
    if row < m && col < n { c[row * n + col] = acc; }
}

// x[r][c] = gelu(x[r][c] + bias[c])
@workgroup(256)
kernel bias_gelu(x: *f32, bias: *f32, rows: u32, cols: u32) {
    let i = global_id.x;
    if i < rows * cols { x[i] = gelu(x[i] + bias[i % cols]); }
}

// Row-wise softmax, one workgroup per row, reductions in shared memory.
@workgroup(256)
kernel softmax(x: *f32, rows: u32, cols: u32) {
    shared red: [256]f32;
    let r = group_id.x;
    let t = local_id.x;
    let base = r * cols;
    var mx = -3.0e38;
    var j = t;
    while j < cols { mx = max(mx, x[base + j]); j += 256; }
    red[t] = mx;
    barrier();
    var s: u32 = 128;
    while s > 0 {
        if t < s { red[t] = max(red[t], red[t + s]); }
        barrier();
        s = s / 2;
    }
    let row_max = red[0];
    barrier();
    var sum = 0.0;
    j = t;
    while j < cols {
        let e = exp(x[base + j] - row_max);
        x[base + j] = e;
        sum += e;
        j += 256;
    }
    red[t] = sum;
    barrier();
    s = 128;
    while s > 0 {
        if t < s { red[t] += red[t + s]; }
        barrier();
        s = s / 2;
    }
    let inv = 1.0 / red[0];
    j = t;
    while j < cols { x[base + j] *= inv; j += 256; }
}

// Row-wise layer norm with learned scale and shift.
@workgroup(256)
kernel layernorm(x: *f32, gamma: *f32, beta: *f32, rows: u32, cols: u32) {
    shared red: [256]f32;
    let r = group_id.x;
    let t = local_id.x;
    let base = r * cols;
    var sum = 0.0;
    var j = t;
    while j < cols { sum += x[base + j]; j += 256; }
    red[t] = sum;
    barrier();
    var s: u32 = 128;
    while s > 0 {
        if t < s { red[t] += red[t + s]; }
        barrier();
        s = s / 2;
    }
    let mean = red[0] / (cols as f32);
    barrier();
    var sq = 0.0;
    j = t;
    while j < cols { let d = x[base + j] - mean; sq += d * d; j += 256; }
    red[t] = sq;
    barrier();
    s = 128;
    while s > 0 {
        if t < s { red[t] += red[t + s]; }
        barrier();
        s = s / 2;
    }
    let inv = rsqrt(red[0] / (cols as f32) + 1.0e-5);
    j = t;
    while j < cols { x[base + j] = (x[base + j] - mean) * inv * gamma[j] + beta[j]; j += 256; }
}

// ---------------------------------------------------------------- CPU versions

export fn matmul_cpu(a: *f32, b: *f32, c: *f32, m: i64, n: i64, k: i64) {
    for i in 0..m {
        let crow = c + i * n;
        for j in 0..n { crow[j] = 0.0; }
        for p in 0..k {
            let av = a[i * k + p];
            let brow = b + p * n;
            for j in 0..n { crow[j] += av * brow[j]; }
        }
    }
}

export fn bias_gelu_cpu(x: *f32, bias: *f32, rows: i64, cols: i64) {
    for r in 0..rows {
        let row = x + r * cols;
        for c in 0..cols { row[c] = gelu(row[c] + bias[c]); }
    }
}

export fn softmax_cpu(x: *f32, rows: i64, cols: i64) {
    for r in 0..rows {
        let row = x + r * cols;
        var mx = -3.0e38;
        for c in 0..cols { mx = max(mx, row[c]); }
        var sum = 0.0;
        for c in 0..cols { let e = exp(row[c] - mx); row[c] = e; sum += e; }
        let inv = 1.0 / sum;
        for c in 0..cols { row[c] *= inv; }
    }
}

export fn layernorm_cpu(x: *f32, gamma: *f32, beta: *f32, rows: i64, cols: i64) {
    for r in 0..rows {
        let row = x + r * cols;
        var sum = 0.0;
        for c in 0..cols { sum += row[c]; }
        let mean = sum / (cols as f32);
        var sq = 0.0;
        for c in 0..cols { let d = row[c] - mean; sq += d * d; }
        let inv = rsqrt(sq / (cols as f32) + 1.0e-5);
        for c in 0..cols { row[c] = (row[c] - mean) * inv * gamma[c] + beta[c]; }
    }
}
