const std = @import("std");

const NUM_COLORS = 3;
const WORD_LEN = 5;
const NUM_CLUES = std.math.powi(u8, NUM_COLORS, WORD_LEN) catch unreachable;
const simd_len = 8; // trial and error. Unclear why this is the best one.

const WordleError = error{
    ShapeMismatch,
    NonContiguous,
};

fn clueEntropiesNoisy(comptime F: type, allocator: std.mem.Allocator, clue_mat_buf: []const u8, clue_mat_index: Index2d, sol_probs: []const F, kernel: []const F) ![]F {
    if (clue_mat_index.cols != sol_probs.len) {
        return error.ShapeMismatch;
    }
    if (clue_mat_index.cstride != 1) {
        debugPrintLn("Clue matrix must be row-contiguous. Found cstride={d}, expected 1", .{clue_mat_index.cstride});
        return error.NonContiguous;
    }
    const num_clues_usize: usize = NUM_CLUES;
    if (kernel.len != num_clues_usize * num_clues_usize) {
        return error.ShapeMismatch;
    }

    const kernel_idx = Index2d.initRowMajor(NUM_CLUES, NUM_CLUES);
    var entropies_ = try allocator.alloc(F, clue_mat_index.rows);
    for (0..clue_mat_index.rows) |i| {
        var weighted_counts = [_]F{0.0} ** NUM_CLUES;
        const row_start = clue_mat_index.linearIndex(i, 0).?;
        const row_end = row_start + clue_mat_index.cols;
        const clue_row = clue_mat_buf[row_start..row_end];
        for (0..clue_mat_index.cols) |j| {
            weighted_counts[clue_row[j]] += sol_probs[j];
        }
        const noisy_clue_prob = try matVecMul(F, allocator, kernel, kernel_idx, &weighted_counts);
        entropies_[i] = entropyBits(F, noisy_clue_prob);
    }
    return entropies_;
}

fn clueEntropies(comptime F: type, allocator: std.mem.Allocator, clue_mat_buf: []const u8, clue_mat_index: Index2d, sol_probs: []const F) ![]F {
    if (clue_mat_index.cols != sol_probs.len) {
        return error.ShapeMismatch;
    }
    if (clue_mat_index.cstride != 1) {
        debugPrintLn("Clue matrix must be row-contiguous. Found cstride={d}, expected 1", .{clue_mat_index.cstride});
        return error.NonContiguous;
    }
    var entropies_ = try allocator.alloc(F, clue_mat_index.rows);
    for (0..clue_mat_index.rows) |i| {
        var weighted_counts = [_]F{0.0} ** NUM_CLUES;
        const row_start = clue_mat_index.linearIndex(i, 0).?;
        const row_end = row_start + clue_mat_index.cols;
        const clue_row = clue_mat_buf[row_start..row_end];
        for (0..clue_mat_index.cols) |j| {
            weighted_counts[clue_row[j]] += sol_probs[j];
        }
        entropies_[i] = entropyBits(F, &weighted_counts);
    }
    return entropies_;
}

fn entropyBits(comptime F: type, dist: []const F) F {
    var h: F = 0.0;
    for (0..dist.len) |i| {
        const p = dist[i];
        if (p > 0)
            h -= p * std.math.log2(p);
    }
    return h;
}

fn noiseKernel(comptime F: type, al: std.mem.Allocator, p_flip: F) ![]F {
    const num_clues_usize: usize = NUM_CLUES;
    var buf = try al.alloc(F, num_clues_usize * num_clues_usize);
    const idx = Index2d.initRowMajor(NUM_CLUES, NUM_CLUES);
    for (0..idx.rows) |i| {
        for (0..idx.cols) |j| {
            const dist = hammingDist(@intCast(i), @intCast(j));
            buf[idx.linearIndex(i, j).?] = transitionProb(F, p_flip, dist);
        }
    }
    return buf;
}

fn transitionProb(comptime F: type, p_flip: F, hamming_dist: u8) F {
    const dist_f: F = @floatFromInt(hamming_dist);
    const p_individual = p_flip / (NUM_COLORS - 1);
    const factor1 = std.math.pow(F, 1 - p_flip, WORD_LEN - dist_f);
    const factor2 = std.math.pow(F, p_individual, dist_f);
    return factor1 * factor2;
}

fn hammingDist(w1: u8, w2: u8) u8 {
    var sum: u8 = 0;
    var remaining1 = w1;
    var remaining2 = w2;
    inline for (0..WORD_LEN) |_| {
        const c1 = remaining1 % NUM_COLORS;
        const c2 = remaining2 % NUM_COLORS;
        if (c1 != c2) {
            sum += 1;
        }
        remaining1 /= NUM_COLORS;
        remaining2 /= NUM_COLORS;
    }
    return sum;
}

fn matVecMul(comptime F: type, al: std.mem.Allocator, mat_buf: []const F, mat_idx: Index2d, vec: []const F) ![]F {
    if (mat_idx.cols != vec.len) {
        return error.ShapeMismatch;
    }
    if (mat_idx.cstride != 1) {
        debugPrintLn("Matrix must be row-contiguous. Found cstride={d}, expected 1", .{mat_idx.cstride});
        return error.NonContiguous;
    }
    var result = try al.alloc(F, vec.len);
    for (0..mat_idx.rows) |i| {
        const row_start = mat_idx.linearIndex(i, 0).?;
        const row_end = row_start + mat_idx.cols;
        const row = mat_buf[row_start..row_end];
        result[i] = dotSimd(F, row, vec);
    }
    return result;
}

fn dot(comptime F: type, v1: []const F, v2: []const F) F {
    var sum: F = 0;
    for (v1, v2) |a, b| {
        sum += a * b;
    }
    return sum;
}

fn dotSimd(comptime F: type, v1: []const F, v2: []const F) F {
    // const simd_len = comptime std.atomic.cache_line / @bitSizeOf(F);
    // std.simd.suggestVectorLengthForCpu(F, comptime cpu: std.Target.Cpu)
    var sum: F = 0;
    var offset: usize = 0;
    while (offset + simd_len <= v1.len) : (offset += simd_len) {
        const v1_simd: @Vector(simd_len, F) = v1[offset..][0..simd_len].*;
        const v2_simd: @Vector(simd_len, F) = v2[offset..][0..simd_len].*;
        sum += @reduce(.Add, v1_simd * v2_simd);
    }
    if (offset < v1.len) {
        for (v1[offset..], v2[offset..]) |a, b| {
            sum += a * b;
        }
    }
    return sum;
}

const Index2d = struct {
    rows: usize,
    cols: usize,
    rstride: usize,
    cstride: usize,
    offset: usize,

    fn linearIndex(self: *const Index2d, i: usize, j: usize) ?usize {
        if (i >= self.rows or j >= self.cols)
            return null;
        return i * self.rstride + j * self.cstride + self.offset;
    }

    fn initRowMajor(rows: usize, cols: usize) Index2d {
        const rstride = cols;
        const cstride = 1;
        return .{ .rows = rows, .cols = cols, .rstride = rstride, .cstride = cstride, .offset = 0 };
    }

    fn initCustomStrides(rows: usize, cols: usize, rstride: usize, cstride: usize) Index2d {
        return .{
            .rows = rows,
            .cols = cols,
            .rstride = rstride,
            .cstride = cstride,
            .offset = 0,
        };
    }
};

fn debugPrintLn(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt ++ "\n", args);
}

fn printLn(comptime fmt: []const u8, args: anytype) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print(fmt ++ "\n", args);
}

test "entropies" {
    const al = std.testing.allocator;

    const clue_mat_buf = [_]u8{
        1, 3, 4, 4,
        0, 4, 0, 0,
        0, 1, 2, 3,
        5, 5, 5, 5,
    };
    const clue_mat_index = Index2d.initRowMajor(4, 4);
    const sol_probs = [_]f64{ 0.25, 0.25, 0.25, 0.25 };

    const expected = [_]f64{ 1.5, -0.75 * std.math.log2(0.75) + 0.5, 2.0, 0.0 };
    const actual = try clueEntropies(f64, al, &clue_mat_buf, clue_mat_index, &sol_probs);
    defer al.free(actual);
    const error_ = error_: {
        var diffs: f64 = 0.0;
        for (0..clue_mat_index.rows) |i| {
            diffs += @abs(expected[i] - actual[i]);
        }
        break :error_ diffs;
    };
    debugPrintLn("Entropies: {d:.3}", .{actual});
    try std.testing.expectApproxEqAbs(0.0, error_, std.math.floatEps(f64));
}

test "noiseKernel" {
    const al = std.testing.allocator;

    const actual = try noiseKernel(f64, al, 0.5);
    defer al.free(actual);
    try std.testing.expectApproxEqAbs(1.0 / 32.0, actual[0], std.math.floatEps(f64));
    try std.testing.expectApproxEqAbs(1.0 / 64.0, actual[1], std.math.floatEps(f64));
    try std.testing.expectApproxEqAbs(1.0 / 64.0, actual[2], std.math.floatEps(f64));
    try std.testing.expectApproxEqAbs(1.0 / 64.0, actual[3], std.math.floatEps(f64));
    try std.testing.expectApproxEqAbs(1.0 / 128.0, actual[4], std.math.floatEps(f64));
    try std.testing.expectApproxEqAbs(1.0 / 64.0, actual[NUM_CLUES], std.math.floatEps(f64));
    try std.testing.expectApproxEqAbs(1.0 / 32.0, actual[NUM_CLUES + 1], std.math.floatEps(f64));
}

test "hammingDist" {
    try std.testing.expectEqual(hammingDist(0, 0), 0);
    try std.testing.expectEqual(hammingDist(0, 1), 1);
    try std.testing.expectEqual(hammingDist(1, 0), 1);
    try std.testing.expectEqual(hammingDist(0, 2), 1);
    try std.testing.expectEqual(hammingDist(0, 3), 1);
    try std.testing.expectEqual(hammingDist(1, 3), 2);
}

test "transitionProb" {
    try std.testing.expectApproxEqAbs(1.0 / 32.0, transitionProb(f64, 0.5, 0), std.math.floatEps(f64));
}

test "matVecMul" {
    const al = std.testing.allocator;

    const clue_mat_buf = [_]f64{
        1, 3, 4, 4,
        0, 4, 0, 0,
        0, 1, 2, 3,
        5, 5, 5, 5,
    };
    const clue_mat_index = Index2d.initRowMajor(4, 4);
    const vec = [_]f64{ -1.0, 0.0, 0.0, 2.0 };

    const expected = [_]f64{ 7.0, 0.0, 6.0, 5.0 };
    const actual = try matVecMul(f64, al, &clue_mat_buf, clue_mat_index, &vec);
    defer al.free(actual);

    const error_ = error_: {
        var diffs: f64 = 0.0;
        for (0..clue_mat_index.rows) |i| {
            diffs += @abs(expected[i] - actual[i]);
        }
        break :error_ diffs;
    };
    debugPrintLn("matVecMul: {d:.2}", .{actual});
    try std.testing.expectApproxEqAbs(0.0, error_, std.math.floatEps(f64));
}

test "Bench 100x100" {
    const al = std.heap.page_allocator;
    const rows = 100;
    const cols = 100;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();

    const tic = std.time.microTimestamp();
    const entropies_ = try clueEntropies(f64, al, inputs_.clues, inputs_.index, inputs_.probs);
    defer al.free(entropies_);
    const toc = std.time.microTimestamp();
    try printLn("100    x    100 took {d}μs", .{toc - tic});
}

test "Bench 100x10_000" {
    const al = std.heap.page_allocator;
    const rows = 100;
    const cols = 10_000;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();

    const tic = std.time.microTimestamp();
    const entropies_ = try clueEntropies(f64, al, inputs_.clues, inputs_.index, inputs_.probs);
    defer al.free(entropies_);
    const toc = std.time.microTimestamp();
    try printLn("100    x 10_000 took {d}μs", .{toc - tic});
}

test "Bench 10_000x1000" {
    const al = std.heap.page_allocator;
    const rows = 10_000;
    const cols = 1000;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropies(f64, al, inputs_.clues, inputs_.index, inputs_.probs);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("10_000 x   1000 took {d}ms", .{toc - tic});
}

test "Bench 10_000x1000 f32" {
    const al = std.heap.page_allocator;
    const rows = 10_000;
    const cols = 1000;

    const inputs_ = try EntropiesInputs32.initRand(al, rows, cols);
    defer inputs_.deinit();

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropies(f32, al, inputs_.clues, inputs_.index, inputs_.probs);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("10_000 x   1000 took {d}ms (f32)", .{toc - tic});
}

test "Bench 1000x10_000" {
    const al = std.heap.page_allocator;
    const rows = 1000;
    const cols = 10_000;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropies(f64, al, inputs_.clues, inputs_.index, inputs_.probs);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("1000   x 10_000 took {d}ms", .{toc - tic});
}

test "Bench 2000x20_000" {
    const al = std.heap.page_allocator;
    const rows = 2000;
    const cols = 20_000;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropies(f64, al, inputs_.clues, inputs_.index, inputs_.probs);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("2_000  x 20_000 took {d}ms", .{toc - tic});
}

test "Bench 10_000x1000 noisy" {
    const al = std.heap.page_allocator;
    const rows = 10_000;
    const cols = 1000;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();
    const kernel = try noiseKernel(f64, al, 0.5);
    defer al.free(kernel);

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropiesNoisy(f64, al, inputs_.clues, inputs_.index, inputs_.probs, kernel);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("10_000 x   1000 took {d}ms (noisy)", .{toc - tic});
}

test "Bench 10_000x1000 noisy f32" {
    const al = std.heap.page_allocator;
    const rows = 10_000;
    const cols = 1000;

    const inputs_ = try EntropiesInputs32.initRand(al, rows, cols);
    defer inputs_.deinit();
    const kernel = try noiseKernel(f32, al, 0.5);
    defer al.free(kernel);

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropiesNoisy(f32, al, inputs_.clues, inputs_.index, inputs_.probs, kernel);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("10_000 x   1000 took {d}ms (noisy, f32)", .{toc - tic});
}

test "Bench 1000x10_000 noisy" {
    const al = std.heap.page_allocator;
    const rows = 1000;
    const cols = 10_000;

    const inputs_ = try EntropiesInputs.initRand(al, rows, cols);
    defer inputs_.deinit();
    const kernel = try noiseKernel(f64, al, 0.5);
    defer al.free(kernel);

    const tic = std.time.milliTimestamp();
    const entropies_ = try clueEntropiesNoisy(f64, al, inputs_.clues, inputs_.index, inputs_.probs, kernel);
    defer al.free(entropies_);
    const toc = std.time.milliTimestamp();
    try printLn("  1000 x 10_000 took {d}ms (noisy)", .{toc - tic});
}

const EntropiesInputs = struct {
    al: std.mem.Allocator,
    index: Index2d,
    clues: []u8,
    probs: []f64,

    fn initRand(al: std.mem.Allocator, rows: usize, cols: usize) !@This() {
        const index = Index2d.initRowMajor(rows, cols);
        var rng = std.Random.DefaultPrng.init(seed: {
            var seed: u64 = undefined;
            try std.posix.getrandom(std.mem.asBytes(&seed));
            break :seed seed;
        });
        const rand = rng.random();
        var clues = try al.alloc(u8, rows * cols);
        rand.bytes(clues);
        for (0..clues.len) |ci| {
            if (clues[ci] >= NUM_CLUES) {
                clues[ci] -= NUM_CLUES;
            }
        }
        const probs = probs: {
            var probs = try al.alloc(f64, cols);
            const cols_f64: f64 = @floatFromInt(cols);
            for (0..cols) |j| {
                probs[j] = 1.0 / cols_f64;
            }
            break :probs probs;
        };
        return .{
            .al = al,
            .clues = clues,
            .index = index,
            .probs = probs,
        };
    }

    fn deinit(self: *const @This()) void {
        self.al.free(self.clues);
        self.al.free(self.probs);
    }
};

const EntropiesInputs32 = struct {
    al: std.mem.Allocator,
    index: Index2d,
    clues: []u8,
    probs: []f32,

    fn initRand(al: std.mem.Allocator, rows: usize, cols: usize) !@This() {
        const index = Index2d.initRowMajor(rows, cols);
        var rng = std.Random.DefaultPrng.init(seed: {
            var seed: u64 = undefined;
            try std.posix.getrandom(std.mem.asBytes(&seed));
            break :seed seed;
        });
        const rand = rng.random();
        var clues = try al.alloc(u8, rows * cols);
        rand.bytes(clues);
        for (0..clues.len) |ci| {
            if (clues[ci] >= NUM_CLUES) {
                clues[ci] -= NUM_CLUES;
            }
        }
        const probs = probs: {
            var probs = try al.alloc(f32, cols);
            const cols_f64: f32 = @floatFromInt(cols);
            for (0..cols) |j| {
                probs[j] = 1.0 / cols_f64;
            }
            break :probs probs;
        };
        return .{
            .al = al,
            .clues = clues,
            .index = index,
            .probs = probs,
        };
    }

    fn deinit(self: *const @This()) void {
        self.al.free(self.clues);
        self.al.free(self.probs);
    }
};
