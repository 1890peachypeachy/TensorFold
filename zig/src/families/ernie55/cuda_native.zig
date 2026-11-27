//! ERNIE 5.5 Flash Preview native CUDA family entry point for tensorfold-native (TP=1/2/4 over dual-rail RoCE + bundled 3-layer DSpark).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const cfg = @import("config.zig");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const ernie_lanes = @import("cuda_lanes.zig");

pub const model_type = "ernie5_5_moe";
pub const formats: []const []const u8 = &.{"fp8-block128"};
pub const default_context: i64 = 262_144;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
pub const window_rows: u32 = engine.window_rows;
pub const explains_open = true;
/// Kept prompt states stand at chunk starts (the HCA rings and compressed blocks line up there).
pub const snap_grid: u32 = engine.snap_grid;

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    rank: u32 = 0,
    tp: u32 = 1,
    master: ?[]const u8 = null,
    master_port: u16 = 29688,
    launch_from: []const u8 = "",
};

pub const Peer = struct { bytes: u64, rank: usize };
pub const Place = struct { rank: usize, world: usize };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    peer_room: ?Peer = null,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
    target: *anyopaque, // the prompt cache keeps this rank group's states (snap_*)
};

/// A state kept at `at`: its bytes on this rank.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return own.e.stateBytes(at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): every rank copies the state there under one new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const own: *Owned = @ptrCast(@alignCast(ptr));
    if (!own.e.keepable(at)) return error.NotAtMark;
    const k = try own.gpa.create(engine.Kept);
    errdefer own.gpa.destroy(k);
    k.* = try own.lanes.x.save(at);
    return k;
}

/// A restore outside a prompt pass would leave the other ranks behind: states resume inside it (Lanes.resumeFrom).
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    own.lanes.x.drop(k.id);
    own.gpa.destroy(k);
}

pub fn place(o: anytype) !Place {
    if (o.tp < 1 or o.rank >= o.tp) return error.NoRank;
    return .{ .rank = o.rank, .world = o.tp };
}

pub fn admit(sm: u32, world: usize) error{ Ernie55RequiresSm121, Ernie55BadTp }!void {
    if (sm != 121) return error.Ernie55RequiresSm121;
    if (world != 1 and world != 2 and world != 4) return error.Ernie55BadTp;
}

/// Per-rank weight share for startup memory admission (~41.5 GiB/Spark at TP=4).
pub fn weightShare(_: std.mem.Allocator, _: std.Io, _: []const u8, total: u64, drafts: bool, _: u32, world: u32) u64 {
    const w = @max(world, 1);
    const est = total / w + (if (drafts) @as(u64, 4) << 30 else @as(u64, 2) << 30);
    return @min(total, est);
}

pub fn capacity(context: usize) usize {
    return ((context + 255) / 256) * 256 + 128;
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    lanes: ernie_lanes.Lanes,
    place: Place,
    lib: ?cuda.nccl.Library = null,
    comm: cuda.nccl.Comm = null,
    group: ?cuda.tp_group.Group = null,
    link: ?follow.Link = null,
    benchmark_group: ?[]const u8 = null,
};

pub fn benchmark(p: *anyopaque) struct { ranks: ?u32, group: ?[]const u8 } {
    const own: *Owned = @ptrCast(@alignCast(p));
    return .{ .ranks = if (own.group) |g| @intCast(g.world) else 1, .group = own.benchmark_group };
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels;
    try cuda.rendezvous.fromMaster(o, cuda.rendezvous.Process{}, cuda.rendezvous.Process{});
    const p = try place(o);
    try admit(@intCast(try ctx.attribute(.compute_capability_major) * 10 + try ctx.attribute(.compute_capability_minor)), p.world);
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    if (p.world > 1) try cuda.tp_watch.start(io, p.rank, p.world);
    if (p.world > 1) std.log.info("[tensorfold] ernie55: rank {d} of {d} ({s})", .{ p.rank, p.world, o.launch_from });
    return openRank(gpa, io, ctx, dir, o, p) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
}

fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: Place) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = undefined, .lanes = undefined, .place = p };

    if (p.world > 1) {
        var lib = try cuda.nccl.Library.open();
        errdefer lib.close();
        var id: cuda.nccl.UniqueId = undefined;
        if (p.rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            cuda.tp_watch.sendId(&id.internal);
        } else {
            id.internal = try cuda.tp_watch.takeId(io);
        }
        var comm: cuda.nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
        own.lib = lib;
        own.comm = comm;
    }

    const cap = capacity(o.context);
    own.e = try engine.Engine.init(gpa, io, ctx.d, dir, p.rank, p.world, .{
        .capacity = cap,
        .load_dspark = o.drafts,
    });
    errdefer own.e.deinit();

    if (p.world > 1) {
        const words_per_row = (own.e.w.config.vocab + 31) / 32;
        const max_link_ids = @max(@max(cap, engine.prompt_rows), engine.window_rows * (words_per_row + 1));
        own.link = try follow.Link.init(gpa, ctx.d, &own.lib.?, own.comm, own.e.stream, max_link_ids);
    }
    errdefer if (own.link) |*l| l.deinit();

    if (p.world > 1) {
        own.group = .{ .lib = own.lib.?, .comm = own.comm, .rank = p.rank, .world = p.world };
        const roce_off = if (std.c.getenv("TF_ROCE")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        if (!roce_off) {
            const max_bytes = try roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null);
            try own.group.?.openRoce(gpa, io, ctx.d, own.e.stream, max_bytes);
        }
        own.e.group = &own.group.?;
    }
    errdefer if (own.group) |*g| if (g.fast) |f| f.deinit();

    // NCCL barrier + room exchange across all ranks (safe across NVMe load skew!)
    var peer: ?Peer = null;
    if (p.world > 1) {
        const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
        var rooms = try cuda.DeviceBuffer.alloc(ctx.d, 8 * p.world);
        defer rooms.free();
        const at = rooms.ptr + p.rank * 8;
        try rooms.upload(p.rank * 8, std.mem.asBytes(&mine));
        try own.lib.?.check(own.lib.?.api.ncclAllGather(at, rooms.ptr, 8, .u8, own.comm, own.e.stream.handle), "ncclAllGather");
        try own.e.stream.synchronize();
        var all: [8]u64 = undefined;
        try rooms.download(0, std.mem.sliceAsBytes(all[0..p.world]));
        for (all[1..p.world], 1..) |r, q| {
            if (peer == null or r < peer.?.bytes) peer = .{ .bytes = r, .rank = q };
        }
        try cuda.tp_watch.ready(io, @splat(0));
    }

    own.benchmark_group = if (own.group) |*g| try core.benchmark_group.cuda(gpa, io, ctx, g, own.e.stream) else null;
    errdefer if (own.benchmark_group) |b| gpa.free(b);

    own.lanes = .{ .x = .{ .e = own.e, .link = if (own.link) |*l| l else null } };
    const sb = own.e.streamBytes();
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = engine.prompt_rows,
        .stream_bytes = sb,
        .free_streams = 1,
        .ctx = own,
        .deinit = close,
        .peer_room = peer,
        .follow = if (p.rank > 0) followRank else null,
        .target = own,
    };
}

pub fn followRank(ptr: *anyopaque) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const link = if (own.link) |*l| l else return;
    return follow.follow(own.e, link);
}

pub const follow_fn = followRank;

fn roceBytes(text: ?[]const u8) !usize {
    const v = text orelse return 64 * 1024 * 1024;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadRoceCap;
    if (n == 0 or n > std.math.maxInt(u32) / 1024) return error.BadRoceCap;
    return n * 1024;
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    cuda.tp_watch.ended();
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    if (own.link) |*l| l.deinit();
    own.e.deinit();
    if (own.group) |*g| if (g.fast) |f| f.deinit();
    if (own.lib) |*lib| {
        _ = lib.api.ncclCommDestroy(own.comm);
        lib.close();
    }
    if (own.benchmark_group) |b| own.gpa.free(b);
    own.gpa.destroy(own);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "ERNIE 5.5 Flash Preview runs at TP 1, 2, or 4: launch with --tp 4 --rank 0..3",
        error.NoCommId => "ERNIE 5.5 TP ranks meet over NCCL: give --master (rank 0's address) or set NCCL_COMM_ID",
        error.Ernie55RequiresSm121 => "ERNIE 5.5 Flash Preview CUDA kernels require GB10 (sm_121)",
        error.Ernie55BadTp => "ERNIE 5.5 Flash Preview supports TP 1, 2, or 4",
        else => null,
    };
}
