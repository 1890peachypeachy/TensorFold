//! TP over processes for ERNIE 5.5: rank 0 broadcasts each step (header + ids) over NCCL; ranks 1..world-1 follow in lockstep.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const sampler = @import("cuda_sampler.zig");
const Engine = engine.Engine;

pub const Kind = enum(u32) {
    begin = 1,
    chunk,
    verify,
    keep,
    block,
    first_masked,
    release,
    idle,
    end,
    save, // a: the kept state's id, seed: its position
    drop, // ids: kept states rank 0 let go
    verify_many, // depth: parts; ids: (slot, n) a part, then every part's ids
    block_many, // depth: parts; ids: (slot, kept_n, anchor, out_len, advance) per part
};

pub const Header = extern struct {
    kind: u32,
    n: u32 = 0,
    depth: u32 = 0,
    flags: u32 = 0,
    a: u64 = 0,
    seed: u64 = 0,
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0,
    slot: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

pub const Words = extern struct { w: [96]u32 };
pub const put_ids = 96 - 16;

pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer,
    ids: []u32,
    gpa: std.mem.Allocator,
    asleep: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, max_ids: usize) !Link {
        const buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        errdefer {
            var owned = buf;
            owned.free();
        }
        return .{
            .d = d,
            .lib = lib,
            .comm = comm,
            .stream = stream,
            .buf = buf,
            .ids = try gpa.alloc(u32, max_ids),
            .gpa = gpa,
        };
    }

    pub fn deinit(l: *Link) void {
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn post(l: *Link, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr, l.buf.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        if (ids.len <= put_ids) {
            var w: Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
            @memcpy(w.w[16..][0..ids.len], ids);
            var a: cuda.Args = .{};
            a.add(l.buf.ptr);
            a.add(@as(u32, @intCast(16 + ids.len)));
            a.add(w);
            try cuda.launch.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.stream, &a);
        } else {
            try l.buf.uploadAsync(0, std.mem.asBytes(&hh), l.stream.handle);
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
        }
        try l.post(64);
        if (ids.len > 0) try l.post(64 + ids.len * 4);
        if (ids.len > put_ids) try l.stream.synchronize();
    }

    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        try l.post(64);
        try l.stream.synchronize();
        var h: Header = undefined;
        try l.buf.download(0, std.mem.asBytes(&h));
        if (h.n > l.ids.len) return error.TooManyIds;
        if (h.n > 0) {
            try l.post(64 + h.n * 4);
            try l.stream.synchronize();
            try l.buf.download(64, std.mem.sliceAsBytes(l.ids[0..h.n]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};

pub const Leader = struct {
    e: *Engine,
    link: ?*Link,

    fn send(x: Leader, h: Header, ids: []const u32) !void {
        const l = x.link orelse return;
        var hh = h;
        hh.slot = x.e.slot;
        if (l.asleep) {
            cuda.tp_watch.wake();
            l.asleep = false;
        }
        if (x.e.dropped.items.len > 0) { // the states rank 0's store let go since the last step, freed first
            const n = @min(x.e.dropped.items.len, l.ids.len);
            try l.send(x.e.k.put_words, .{ .kind = @intFromEnum(Kind.drop), .slot = x.e.slot }, x.e.dropped.items[0..n]);
            x.e.dropped.replaceRangeAssumeCapacity(0, n, &.{});
        }
        try l.send(x.e.k.put_words, hh, ids);
    }

    /// The bound sequence's state at `at` kept on every rank under one new id.
    pub fn save(x: Leader, at: usize) !engine.Kept {
        const id = x.e.next_snap;
        x.e.next_snap += 1; // an id is used once, even when this rank's copy then fails
        if (at == 0 or at != x.e.pos()) return error.BadKeep;
        try x.send(.{ .kind = @intFromEnum(Kind.save), .a = id, .seed = at }, &.{});
        x.e.keepAs(id, at) catch |err| {
            x.drop(id); // the other ranks free their copy at the next step
            return err;
        };
        return .{ .id = id, .at = @intCast(at) };
    }

    /// A state rank 0's store let go: freed here now, on the other ranks at the next step.
    pub fn drop(x: Leader, id: u64) void {
        x.e.dropState(id);
        if (x.link != null) x.e.dropped.append(x.e.gpa, @intCast(id)) catch {};
    }

    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    pub fn begin(x: Leader, sampling: ?sampler.Sampling) !void {
        const s = sampler.check(sampling);
        var h: Header = .{ .kind = @intFromEnum(Kind.begin) };
        if (s) |v| {
            h.flags = 1;
            h.seed = v.seed;
            h.temperature = v.temperature;
            h.top_p = v.top_p;
            h.min_p = v.min_p;
            h.top_k = v.top_k;
        }
        try x.send(h, &.{});
        try x.e.begin(s);
    }

    pub fn chunk(x: Leader, ids: []const u32, pos0: usize, total: usize, is_last: bool, resume_id: u64) !u32 {
        try x.send(.{
            .kind = @intFromEnum(Kind.chunk),
            .depth = @intCast(pos0),
            .flags = if (is_last) 1 else 0,
            .a = total,
            .seed = resume_id,
        }, ids);
        return x.e.chunk(ids, pos0, total, is_last, resume_id);
    }

    pub fn firstMasked(x: Leader, bits: u64, masks: []const u32) !u32 {
        try x.send(.{
            .kind = @intFromEnum(Kind.first_masked),
            .depth = if (masks.len > 0) 1 else 0,
        }, masks);
        return x.e.firstMasked(bits);
    }

    pub fn verify(x: Leader, ids: []const u32, out: []u32, bits: u64, masks: []const u32) !void {
        if (masks.len == 0) {
            try x.send(.{ .kind = @intFromEnum(Kind.verify), .depth = 0 }, ids);
            return x.e.verify(ids, out, 0);
        }
        const total = ids.len + masks.len;
        const buf = try x.e.gpa.alloc(u32, total);
        defer x.e.gpa.free(buf);
        @memcpy(buf[0..ids.len], ids);
        @memcpy(buf[ids.len..], masks);
        try x.send(.{ .kind = @intFromEnum(Kind.verify), .depth = @intCast(ids.len) }, buf);
        return x.e.verify(ids, out, bits);
    }

    pub fn verifyMany(x: Leader, parts: []const engine.Part) !void {
        var buf: [2 * engine.max_parts + engine.window_rows]u32 = undefined;
        var at: usize = 2 * parts.len;
        for (parts, 0..) |pt, i| {
            buf[2 * i] = pt.slot;
            buf[2 * i + 1] = @intCast(pt.ids.len);
            @memcpy(buf[at..][0..pt.ids.len], pt.ids);
            at += pt.ids.len;
        }
        try x.send(.{ .kind = @intFromEnum(Kind.verify_many), .depth = @intCast(parts.len) }, buf[0..at]);
        return x.e.verifyMany(parts);
    }

    pub fn keep(x: Leader, n: usize) !void {
        try x.send(.{ .kind = @intFromEnum(Kind.keep), .a = n }, &.{});
        try x.e.keep(n);
    }

    pub fn block(x: Leader, kept_n: usize, anchor: u32, out: []u32) !void {
        try x.send(.{
            .kind = @intFromEnum(Kind.block),
            .depth = @intCast(out.len),
            .flags = anchor,
            .a = kept_n,
        }, &.{});
        try x.e.block(kept_n, anchor, out);
    }

    pub fn blockMany(x: Leader, parts: []const engine.BlockPart) !void {
        var buf: [engine.max_parts * 5]u32 = undefined;
        for (parts, 0..) |pt, i| {
            buf[5 * i + 0] = pt.slot;
            buf[5 * i + 1] = @intCast(pt.kept_n);
            buf[5 * i + 2] = pt.anchor;
            buf[5 * i + 3] = @intCast(pt.out.len);
            buf[5 * i + 4] = @intCast(pt.advance);
        }
        try x.send(.{ .kind = @intFromEnum(Kind.block_many), .depth = @intCast(parts.len) }, buf[0 .. parts.len * 5]);
        try x.e.blockMany(parts);
    }

    pub fn release(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
        try x.send(.{ .kind = @intFromEnum(Kind.release) }, &.{});
        x.e.release(slot);
    }

    pub fn idle(x: Leader) !void {
        const l = x.link orelse return;
        if (l.asleep) return;
        try x.send(.{ .kind = @intFromEnum(Kind.idle) }, &.{});
        try l.stream.synchronize();
        l.asleep = true;
    }

    pub fn end(x: Leader) !void {
        try x.send(.{ .kind = @intFromEnum(Kind.end) }, &.{});
    }
};

pub fn follow(e: *Engine, link: *Link) !void {
    const words_per_row = (e.w.config.vocab + 31) / 32;
    var mask_dev: ?cuda.DeviceBuffer = null;
    defer if (mask_dev) |*b| b.free();
    var out_buf: [engine.window_rows]u32 = undefined;

    while (true) {
        const msg = try link.receive();
        const h = msg.h;
        const kind: Kind = @enumFromInt(h.kind);
        switch (kind) {
            .end => return,
            .idle => cuda.tp_watch.sleep(e.io),
            .release => e.release(h.slot),
            .begin => {
                try e.bind(h.slot);
                const s: ?sampler.Sampling = if (h.flags != 0) .{
                    .seed = h.seed,
                    .temperature = h.temperature,
                    .top_p = h.top_p,
                    .min_p = h.min_p,
                    .top_k = h.top_k,
                } else null;
                try e.begin(s);
            },
            .chunk => {
                try e.bind(h.slot);
                _ = try e.chunk(msg.ids, h.depth, @intCast(h.a), h.flags != 0, h.seed);
            },
            .first_masked => {
                try e.bind(h.slot);
                var bits: u64 = 0;
                if (h.depth != 0 and msg.ids.len == words_per_row) {
                    if (mask_dev == null) mask_dev = try cuda.DeviceBuffer.alloc(e.d, engine.window_rows * words_per_row * 4);
                    try mask_dev.?.upload(0, std.mem.sliceAsBytes(msg.ids));
                    bits = mask_dev.?.ptr;
                }
                _ = try e.firstMasked(bits);
            },
            .verify => {
                try e.bind(h.slot);
                if (h.depth == 0) {
                    try e.verify(msg.ids, out_buf[0..msg.ids.len], 0);
                } else {
                    const n_ids: usize = h.depth;
                    const masks = msg.ids[n_ids..];
                    if (mask_dev == null) mask_dev = try cuda.DeviceBuffer.alloc(e.d, engine.window_rows * words_per_row * 4);
                    try mask_dev.?.upload(0, std.mem.sliceAsBytes(masks));
                    try e.verify(msg.ids[0..n_ids], out_buf[0..n_ids], mask_dev.?.ptr);
                }
            },
            .keep => {
                try e.bind(h.slot);
                try e.keep(@intCast(h.a));
            },
            .block => {
                try e.bind(h.slot);
                try e.block(@intCast(h.a), h.flags, out_buf[0..h.depth]);
            },
            .save => {
                try e.bind(h.slot);
                try e.keepAs(h.a, @intCast(h.seed));
            },
            .drop => for (msg.ids) |id| e.dropState(id),
            .verify_many => {
                const np: usize = h.depth;
                if (np == 0 or np > engine.max_parts) return error.BadParts;
                var parts: [engine.max_parts]engine.Part = undefined;
                var outs: [engine.window_rows]u32 = undefined;
                var at: usize = 2 * np;
                var row: usize = 0;
                for (0..np) |i| {
                    const n: usize = msg.ids[2 * i + 1];
                    if (row + n > engine.window_rows or at + n > msg.ids.len) return error.BadParts;
                    parts[i] = .{ .slot = msg.ids[2 * i], .ids = msg.ids[at..][0..n], .out = outs[row..][0..n] };
                    at += n;
                    row += n;
                }
                try e.verifyMany(parts[0..np]);
            },
            .block_many => {
                const np: usize = h.depth;
                if (np == 0 or np > engine.max_parts or msg.ids.len < np * 5) return error.BadParts;
                var parts: [engine.max_parts]engine.BlockPart = undefined;
                var outs: [engine.max_parts * 7]u32 = undefined;
                for (0..np) |i| {
                    const out_len: usize = msg.ids[5 * i + 3];
                    if (out_len > 7) return error.BadParts;
                    parts[i] = .{
                        .slot = msg.ids[5 * i + 0],
                        .kept_n = msg.ids[5 * i + 1],
                        .anchor = msg.ids[5 * i + 2],
                        .out = outs[i * 7 ..][0..out_len],
                        .advance = msg.ids[5 * i + 4],
                    };
                }
                try e.blockMany(parts[0..np]);
            },
        }
    }
}
