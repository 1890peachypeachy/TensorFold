//! tf-ernie55-cuda: standalone parity check & decode/prefill/DSpark benchmark for ERNIE 5.5 Flash Preview on CUDA (TP=1/2/4).
const std = @import("std");
const cuda = @import("cuda");
const ernie = @import("ernie55");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);

    var model_dir: []const u8 = "/var/tmp/models/ERNIE-5.5-Flash-Preview";
    var max_layers: ?u32 = null;
    var decode_steps: usize = 16;
    var dspark_k: usize = 0;
    var tp: usize = 1;
    var rank: usize = 0;
    var capacity: usize = 4096;
    var prompt_str: []const u8 = "1,20000,1500,500";
    var dump_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
            i += 1;
            max_layers = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--steps") and i + 1 < args.len) {
            i += 1;
            decode_steps = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--dspark") and i + 1 < args.len) {
            i += 1;
            dspark_k = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--tp") and i + 1 < args.len) {
            i += 1;
            tp = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--rank") and i + 1 < args.len) {
            i += 1;
            rank = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
            i += 1;
            capacity = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--ids") and i + 1 < args.len) {
            i += 1;
            prompt_str = args[i];
        } else if (std.mem.eql(u8, args[i], "--dump") and i + 1 < args.len) {
            i += 1;
            dump_path = args[i];
        }
    }

    var prompt_ids: std.ArrayList(i32) = .empty;
    var it = std.mem.splitScalar(u8, prompt_str, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        try prompt_ids.append(a, try std.fmt.parseInt(i32, tok, 10));
    }
    const n_prompt = prompt_ids.items.len;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var comm_stream = try cuda.Stream.init(&driver, true);
    defer comm_stream.deinit();
    var ev_shared = try cuda.Event.init(&driver, false);
    defer ev_shared.deinit();
    var ev_comm = try cuda.Event.init(&driver, false);
    defer ev_comm.deinit();

    var group_opt: ?cuda.tp_group.Group = null;
    if (tp > 1) {
        group_opt = try cuda.tp_group.Group.open(&driver, rank, tp);
        if (init.environ_map.get("TF_ROCE") != null) try group_opt.?.openRoce(gpa, io, &driver, stream, 64 * 1024 * 1024);
    }
    defer if (group_opt) |*g| g.close();

    var lane = try cuda.qmmf.Lane.load(&driver, 12);
    defer lane.unload();
    lane.pdl = true;

    var kernels = try ernie.cuda_kernels.Kernels.load(&driver);
    defer kernels.unload();

    const t0 = std.Io.Clock.awake.now(io);
    var w = try ernie.cuda_weights.load(gpa, io, &driver, model_dir, rank, tp, .{
        .max_layers = max_layers,
        .load_dspark = dspark_k > 0,
    });
    defer w.deinit();
    const t_load = std.Io.Clock.awake.now(io);
    const load_s = @as(f64, @floatFromInt(t0.durationTo(t_load).nanoseconds)) / 1e9;
    if (rank == 0) {
        std.debug.print("[ernie55] loaded {d} layers (dspark={d}) in {d:.2}s ({d:.2} GiB)\n", .{
            w.loaded_layers, dspark_k, load_s, @as(f64, @floatFromInt(w.bytes)) / (1024.0 * 1024.0 * 1024.0),
        });
    }

    // NCCL barrier after weight load so slower NVMe ranks finish before any RoCE fast-path spin-wait.
    if (group_opt) |*g| {
        var bar: [8]u64 = @splat(0);
        bar[rank] = 0xE55_0001;
        try g.gather(u64, &driver, stream, 0xE55_0001, bar[0..tp]);
    }

    var kvc = try ernie.cuda_kv.KvCache.alloc(gpa, &driver, &w.config, w.loaded_layers, capacity);
    defer kvc.deinit();
    const max_rows = @max(n_prompt, @max(@as(usize, 16), dspark_k + 1));
    var sc = try ernie.cuda_kv.Scratch.alloc(&driver, &w.config, &w.share, max_rows, capacity);
    defer sc.deinit();

    const runner: ernie.cuda_forward.Runner = .{
        .k = &kernels,
        .lane = &lane,
        .w = &w,
        .group = if (group_opt) |*g| g else null,
        .comm_stream = comm_stream,
        .ev_shared = ev_shared,
        .ev_comm = ev_comm,
    };

    var dspark_caches: ?ernie.cuda_dspark.Caches = null;
    defer if (dspark_caches) |*dc| dc.deinit();
    var drafter: ?ernie.cuda_dspark.Drafter = null;
    defer if (drafter) |*dr| dr.deinit();
    if (dspark_k > 0) {
        dspark_caches = try ernie.cuda_dspark.Caches.init(&driver);
        drafter = try ernie.cuda_dspark.Drafter.init(&driver, &w.dspark.?, w.share.vocab_width);
    }

    // Helper: pick greedy argmax across TP ranks for `n_rows` rows using 1 single RoCE allGather.
    const Pair16 = extern struct { val: f32, id: i32, pad0: u32 = 0, pad1: u32 = 0 };
    const pickRows = struct {
        fn run(d: *const cuda.Driver, s: cuda.Stream, g_opt: ?*const cuda.tp_group.Group, sc_ptr: *const ernie.cuda_kv.Scratch, row0: usize, n_rows: usize, out: []i32) !void {
            var ids_loc: [16]i32 = undefined;
            var vals_loc: [16]f32 = undefined;
            try cuda.memory.download(d, sc_ptr.argmax_id + row0 * 4, std.mem.sliceAsBytes(ids_loc[0..n_rows]));
            try cuda.memory.download(d, sc_ptr.argmax_val + row0 * 4, std.mem.sliceAsBytes(vals_loc[0..n_rows]));
            if (g_opt) |g| {
                if (g.world > 1) {
                    var loc: [16]Pair16 = undefined;
                    for (0..n_rows) |r| loc[r] = .{ .val = vals_loc[r], .id = ids_loc[r] };
                    const bytes = n_rows * @sizeOf(Pair16);
                    try sc_ptr.buf.upload(sc_ptr.moe_pack - sc_ptr.buf.ptr, std.mem.sliceAsBytes(loc[0..n_rows]));
                    try g.allGather(s, sc_ptr.moe_pack, sc_ptr.tp_gather, bytes);
                    try s.synchronize();
                    try g.check();
                    var all: [8 * 16]Pair16 = undefined;
                    try cuda.memory.download(d, sc_ptr.tp_gather, std.mem.sliceAsBytes(all[0 .. g.world * n_rows]));
                    for (0..n_rows) |r| {
                        var best = all[r];
                        for (1..g.world) |wk| {
                            const p = all[wk * n_rows + r];
                            if (p.val > best.val or (p.val == best.val and p.id < best.id)) best = p;
                        }
                        out[r] = best.id;
                    }
                    return;
                }
            }
            for (0..n_rows) |r| out[r] = ids_loc[r];
        }
    }.run;

    // Prefill
    const pos_host = try a.alloc(i32, n_prompt);
    for (0..n_prompt) |idx| pos_host[idx] = @intCast(idx);
    try sc.buf.upload(sc.ids - sc.buf.ptr, std.mem.sliceAsBytes(prompt_ids.items));
    try sc.buf.upload(sc.positions - sc.buf.ptr, std.mem.sliceAsBytes(pos_host));

    const t_pf0 = std.Io.Clock.awake.now(io);
    const pf_taps: u64 = if (dspark_k > 0) sc.dspark_taps else 0;
    try runner.forwardWithTaps(stream, &kvc, &sc, n_prompt, n_prompt - 1, true, pf_taps);
    if (dspark_k > 0) {
        try drafter.?.absorb(&runner, stream, &dspark_caches.?, &sc, sc.dspark_taps, n_prompt, 0);
    }
    try stream.synchronize();
    if (group_opt) |*g| try g.check();
    const t_pf1 = std.Io.Clock.awake.now(io);
    const pf_ms = @as(f64, @floatFromInt(t_pf0.durationTo(t_pf1).nanoseconds)) / 1e6;

    if (dump_path) |dp| {
        if (rank == 0) {
            const h_bytes = try gpa.alloc(u8, n_prompt * w.config.hidden * 2);
            defer gpa.free(h_bytes);
            try cuda.memory.download(&driver, sc.layer_in, h_bytes);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dp, .data = h_bytes });
            std.debug.print("[ernie55] dumped final hidden ({d} bytes) -> {s}\n", .{ h_bytes.len, dp });
        }
    }

    var first_arr: [1]i32 = undefined;
    try pickRows(&driver, stream, if (group_opt) |*g| g else null, &sc, n_prompt - 1, 1, &first_arr);
    var next_tok = first_arr[0];
    var gen = try a.alloc(i32, decode_steps);
    var cur_pos: usize = n_prompt;
    const use_graph = (tp == 1) or (if (group_opt) |*g| g.fast != null else false);

    if (dspark_k == 0) {
        // Plain 1-token decode with CUDA Graph
        var graph_exec: ?cuda.graph.Exec = null;
        defer if (graph_exec) |*ge| ge.deinit();
        var t_dec0 = std.Io.Clock.awake.now(io);
        var timed_steps: usize = 0;
        for (0..decode_steps) |step| {
            if (step == 2) t_dec0 = std.Io.Clock.awake.now(io);
            if (step >= 2 and step + 1 < decode_steps) timed_steps += 1;
            gen[step] = next_tok;
            if (step + 1 == decode_steps) break;
            const pos_i32: i32 = @intCast(cur_pos);
            try sc.buf.upload(sc.ids - sc.buf.ptr, std.mem.asBytes(&next_tok));
            try sc.buf.upload(sc.positions - sc.buf.ptr, std.mem.asBytes(&pos_i32));
            if (use_graph and cur_pos + 1 <= 2176) {
                if (graph_exec) |ge| {
                    try ge.launchOn(stream);
                } else if (step == 0) {
                    try runner.forward(stream, &kvc, &sc, 1, cur_pos, true);
                } else {
                    try cuda.graph.beginCapture(stream, .thread_local);
                    try runner.forward(stream, &kvc, &sc, 1, cur_pos, true);
                    var g = try cuda.graph.endCapture(stream);
                    defer g.deinit();
                    graph_exec = try g.instantiate();
                    try graph_exec.?.launchOn(stream);
                }
            } else {
                try runner.forward(stream, &kvc, &sc, 1, cur_pos, true);
            }
            try stream.synchronize();
            if (group_opt) |*g| try g.check();
            try pickRows(&driver, stream, if (group_opt) |*g| g else null, &sc, 0, 1, &first_arr);
            next_tok = first_arr[0];
            cur_pos += 1;
        }
        const t_dec1 = std.Io.Clock.awake.now(io);
        const dec_ms = @as(f64, @floatFromInt(t_dec0.durationTo(t_dec1).nanoseconds)) / 1e6;
        if (rank == 0) {
            const steps_run = if (timed_steps > 0) timed_steps else 1;
            const ms_per_tok = dec_ms / @as(f64, @floatFromInt(steps_run));
            const tok_s = 1000.0 / ms_per_tok;
            std.debug.print("[ernie55] PLAIN layers={d} tp={d} weights={d:.2} GiB | prefill({d} tok)={d:.2} ms | decode({d} tok)={d:.2} ms ({d:.2} ms/tok, {d:.2} tok/s)\n", .{
                w.loaded_layers, tp, @as(f64, @floatFromInt(w.bytes)) / (1024.0 * 1024.0 * 1024.0), n_prompt, pf_ms, steps_run, dec_ms, ms_per_tok, tok_s,
            });
            std.debug.print("[ernie55] generated ids:", .{});
            for (gen) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    } else {
        // Bundled 3-layer DSpark speculative decoding with CUDA Graph on the (K+1)-row target verify step!
        const R = dspark_k + 1;
        var verify_graph: ?cuda.graph.Exec = null;
        defer if (verify_graph) |*ge| ge.deinit();

        var produced: usize = 0;
        var rounds: usize = 0;
        var timed_rounds: usize = 0;
        var timed_tokens: usize = 0;
        var draft_ns_sum: i128 = 0;
        var verify_ns_sum: i128 = 0;
        var absorb_ns_sum: i128 = 0;
        var t_dec0 = std.Io.Clock.awake.now(io);

        var drafts: [8]i32 = undefined;
        var verify_ids: [8]i32 = undefined;
        var verify_pos: [8]i32 = undefined;
        var targ_out: [8]i32 = undefined;

        while (produced < decode_steps) {
            if (rounds == 2) t_dec0 = std.Io.Clock.awake.now(io);

            // 1. DSpark draft K tokens from anchor = next_tok at position cur_pos
            const td0 = std.Io.Clock.awake.now(io);
            try drafter.?.draft(&runner, stream, &dspark_caches.?, &sc, next_tok, cur_pos, drafts[0..dspark_k]);
            const td1 = std.Io.Clock.awake.now(io);

            // 2. Target verify R = K + 1 tokens: [next_tok, drafts[0..K]] at [cur_pos .. cur_pos + K]
            verify_ids[0] = next_tok;
            verify_pos[0] = @intCast(cur_pos);
            for (0..dspark_k) |j| {
                verify_ids[j + 1] = drafts[j];
                verify_pos[j + 1] = @intCast(cur_pos + j + 1);
            }
            const tv0 = std.Io.Clock.awake.now(io);
            try sc.buf.upload(sc.ids - sc.buf.ptr, std.mem.sliceAsBytes(verify_ids[0..R]));
            try sc.buf.upload(sc.positions - sc.buf.ptr, std.mem.sliceAsBytes(verify_pos[0..R]));
            if (use_graph and cur_pos + R <= 2176) {
                if (verify_graph) |ge| {
                    try ge.launchOn(stream);
                } else if (rounds == 0) {
                    try runner.forwardWithTaps(stream, &kvc, &sc, R, cur_pos + dspark_k, true, sc.dspark_taps);
                } else {
                    try cuda.graph.beginCapture(stream, .thread_local);
                    try runner.forwardWithTaps(stream, &kvc, &sc, R, cur_pos + dspark_k, true, sc.dspark_taps);
                    var g = try cuda.graph.endCapture(stream);
                    defer g.deinit();
                    verify_graph = try g.instantiate();
                    try verify_graph.?.launchOn(stream);
                }
            } else {
                try runner.forwardWithTaps(stream, &kvc, &sc, R, cur_pos + dspark_k, true, sc.dspark_taps);
            }
            try stream.synchronize();
            if (group_opt) |*g| try g.check();
            try pickRows(&driver, stream, if (group_opt) |*g| g else null, &sc, 0, R, targ_out[0..R]);
            const tv1 = std.Io.Clock.awake.now(io);

            // 3. Commit accepted tokens (row 0 = next_tok always committed, plus matching drafts)
            var accepted: usize = 1;
            if (produced < decode_steps) {
                gen[produced] = next_tok;
                produced += 1;
            }
            for (0..dspark_k) |j| {
                if (targ_out[j] == drafts[j]) {
                    if (produced < decode_steps) {
                        gen[produced] = drafts[j];
                        produced += 1;
                    }
                    accepted += 1;
                } else {
                    break;
                }
            }
            const new_next = targ_out[accepted - 1];

            // 4. Absorb ONLY the `accepted` committed rows into DSpark context KV
            const ta0 = std.Io.Clock.awake.now(io);
            try drafter.?.absorb(&runner, stream, &dspark_caches.?, &sc, sc.dspark_taps, accepted, cur_pos);
            try stream.synchronize();
            const ta1 = std.Io.Clock.awake.now(io);

            if (rounds >= 2) {
                timed_rounds += 1;
                timed_tokens += accepted;
                draft_ns_sum += td0.durationTo(td1).nanoseconds;
                verify_ns_sum += tv0.durationTo(tv1).nanoseconds;
                absorb_ns_sum += ta0.durationTo(ta1).nanoseconds;
            }
            cur_pos += accepted;
            next_tok = new_next;
            rounds += 1;
        }
        const t_dec1 = std.Io.Clock.awake.now(io);
        const dec_ms = @as(f64, @floatFromInt(t_dec0.durationTo(t_dec1).nanoseconds)) / 1e6;
        if (rank == 0) {
            const tr: f64 = @floatFromInt(@max(timed_rounds, 1));
            const tt: f64 = @floatFromInt(@max(timed_tokens, 1));
            const tok_s = tt * 1000.0 / dec_ms;
            const mean_acc = tt / tr;
            const d_ms = @as(f64, @floatFromInt(draft_ns_sum)) / (tr * 1e6);
            const v_ms = @as(f64, @floatFromInt(verify_ns_sum)) / (tr * 1e6);
            const a_ms = @as(f64, @floatFromInt(absorb_ns_sum)) / (tr * 1e6);
            std.debug.print("[ernie55] DSPARK(k={d}) layers={d} tp={d} weights={d:.2} GiB | prefill({d} tok)={d:.2} ms | {d} tok / {d} rnds (acc={d:.2} tok/rnd) in {d:.2} ms ({d:.2} tok/s) | draft={d:.2}ms verify={d:.2}ms absorb={d:.2}ms\n", .{
                dspark_k, w.loaded_layers, tp, @as(f64, @floatFromInt(w.bytes)) / (1024.0 * 1024.0 * 1024.0), n_prompt, pf_ms, timed_tokens, timed_rounds, mean_acc, dec_ms, tok_s, d_ms, v_ms, a_ms,
            });
            std.debug.print("[ernie55] generated ids:", .{});
            for (gen) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    }
    return 0;
}
