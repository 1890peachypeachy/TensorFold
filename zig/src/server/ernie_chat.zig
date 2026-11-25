//! Native ERNIE 5.5 Flash Preview chat renderer (matching checkpoint's encoding_ernie.py).
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Span = model_text.Span;
const Segment = model_text.Segment;

pub const START = "<|ernie:im_start|>";
pub const END = "<|ernie:im_end|>";
pub const THINK = "<ernie:think>";
pub const THINK_END = "</ernie:think>";
pub const CALL = "<ernie:tool_call>";
pub const CALL_END = "</ernie:tool_call>";
pub const OUTPUT = "<ernie:tool_output>";
pub const OUTPUT_END = "</ernie:tool_output>";

const Builder = struct {
    a: Allocator,
    text: std.ArrayList(u8) = .empty,
    spans: std.ArrayList(Span) = .empty,
    segs: std.ArrayList(Segment) = .empty,

    fn control(b: *Builder, s: []const u8) !void {
        if (s.len == 0) return;
        try b.text.appendSlice(b.a, s);
        if (b.segs.items.len > 0 and b.segs.items[b.segs.items.len - 1].allow_special) {
            b.segs.items[b.segs.items.len - 1].end = b.text.items.len;
        } else {
            try b.segs.append(b.a, .{ .end = b.text.items.len, .allow_special = true });
        }
    }

    fn plain(b: *Builder, s: []const u8) !void {
        if (s.len == 0) return;
        const start = b.text.items.len;
        try b.text.appendSlice(b.a, s);
        const end = b.text.items.len;
        if (b.spans.items.len > 0 and b.spans.items[b.spans.items.len - 1][1] == start) {
            b.spans.items[b.spans.items.len - 1][1] = end;
        } else {
            try b.spans.append(b.a, .{ start, end });
        }
        if (b.segs.items.len > 0 and !b.segs.items[b.segs.items.len - 1].allow_special) {
            b.segs.items[b.segs.items.len - 1].end = end;
        } else {
            try b.segs.append(b.a, .{ .end = end, .allow_special = false });
        }
    }
};

fn renderContent(b: *Builder, content: ?Value) !void {
    const c = content orelse return;
    switch (c) {
        .string => |s| try b.plain(s),
        .array => |arr| {
            for (arr) |item| {
                if (item != .object) continue;
                const kind = item.get("type") orelse continue;
                if (kind == .string and std.mem.eql(u8, kind.string, "text")) {
                    if (item.get("text")) |tv| {
                        if (tv == .string) try b.plain(tv.string);
                    }
                }
            }
        },
        else => {},
    }
}

fn contentText(a: Allocator, content: ?Value) ![]const u8 {
    const c = content orelse return "";
    return switch (c) {
        .string => |s| s,
        .array => |arr| blk: {
            var out: std.ArrayList(u8) = .empty;
            for (arr) |item| {
                if (item != .object) continue;
                const kind = item.get("type") orelse continue;
                if (kind == .string and std.mem.eql(u8, kind.string, "text")) {
                    if (item.get("text")) |tv| {
                        if (tv == .string) try out.appendSlice(a, tv.string);
                    }
                }
            }
            break :blk out.items;
        },
        else => "",
    };
}

fn renderAssistant(b: *Builder, msg: Value, thinking_off: bool) !void {
    const full = try contentText(b.a, msg.get("content"));
    var reasoning: []const u8 = "";
    var body_text: []const u8 = full;

    if (msg.get("reasoning_content")) |rc| {
        if (rc == .string) reasoning = rc.string;
    } else if (msg.get("thoughts")) |th| {
        if (th == .string) reasoning = th.string;
    } else if (std.mem.indexOf(u8, full, THINK_END)) |first_end| {
        const start = if (std.mem.lastIndexOf(u8, full[0..first_end], THINK)) |s| s + THINK.len else 0;
        reasoning = full[start..first_end];
        const last_end = std.mem.lastIndexOf(u8, full, THINK_END).? + THINK_END.len;
        body_text = full[last_end..];
    }

    try b.control(START ++ "assistant\n" ++ THINK);
    if (!thinking_off and std.mem.trim(u8, reasoning, " \t\r\n").len > 0) {
        try b.plain(reasoning);
    }
    try b.control(THINK_END);
    try b.plain(body_text);

    if (msg.get("tool_calls")) |tc| {
        if (tc == .array and tc.array.len > 0) {
            try b.control(CALL ++ "\n");
            for (tc.array) |call| {
                if (call != .object) continue;
                const func = call.get("function") orelse call;
                if (func != .object) continue;
                const name_v = func.get("name") orelse continue;
                if (name_v != .string) continue;
                try b.control("<invoke name=\"");
                try b.plain(name_v.string);
                try b.control("\">");
                if (func.get("arguments")) |args_raw| {
                    const args_obj: ?Value = switch (args_raw) {
                        .object => args_raw,
                        .string => |s| switch (try json.parse(b.a, s)) {
                            .ok => |v| if (v == .object) v else null,
                            .err => null,
                        },
                        else => null,
                    };
                    if (args_obj) |ao| {
                        for (ao.object.keys(), ao.object.values()) |k, v| {
                            try b.control("<parameter name=\"");
                            try b.plain(k);
                            try b.control("\">");
                            if (v == .string) {
                                try b.plain(v.string);
                            } else {
                                const js = try json.stringify(b.a, v, .{});
                                try b.plain(js);
                            }
                            try b.control("</parameter>");
                        }
                    }
                }
                try b.control("</invoke>\n");
            }
            try b.control(CALL_END);
        }
    }
    try b.control(END ++ "\n");
}

pub fn render(a: Allocator, messages: Value, options: model_text.RenderOptions, problem: *[]const u8) model_text.Error!model_text.Rendered {
    if (messages != .array or messages.array.len == 0) {
        problem.* = "messages must be a non-empty list";
        return error.Template;
    }
    const msgs = messages.array;
    const thinking_off = !options.enable_thinking;
    const effort: []const u8 = if (options.reasoning_effort) |e| blk: {
        if (std.mem.eql(u8, e, "low") or std.mem.eql(u8, e, "medium") or std.mem.eql(u8, e, "high")) break :blk e;
        break :blk "high";
    } else "high";

    const tools: []const Value = if (options.tools) |tv| (if (tv == .array) tv.array else &.{}) else &.{};

    var b: Builder = .{ .a = a };

    // System header block (matches encoding_ernie.py)
    try b.control(START ++ "system\n");
    const has_leading_sys = msgs[0] == .object and if (msgs[0].get("role")) |r| (r == .string and std.mem.eql(u8, r.string, "system")) else false;
    if (has_leading_sys) {
        try b.control("<ernie:system_setting>\n");
        try renderContent(&b, msgs[0].get("content"));
        try b.control("\n</ernie:system_setting>\n");
    }
    try b.control("<ernie:global_setting>");
    if (!thinking_off) {
        try b.control("\nreasoning_effort=");
        try b.plain(effort);
    }
    if (tools.len > 0) {
        try b.control("\nparallel_tool_calls=True");
    }
    try b.control("\n</ernie:global_setting>");
    if (tools.len > 0) {
        try b.control("\n<ernie:tool_list>\n[");
        for (tools, 0..) |t, ti| {
            if (ti > 0) try b.plain(", ");
            const js = try json.stringify(a, t, .{});
            try b.plain(js);
        }
        try b.control("]\n</ernie:tool_list>");
    }
    try b.control(END ++ "\n");

    var starts_list: std.ArrayList(usize) = .empty;
    for (msgs, 0..) |msg, idx| {
        if (msg != .object) continue;
        const role_v = msg.get("role") orelse continue;
        if (role_v != .string) continue;
        const role = role_v.string;
        if (idx == 0 and std.mem.eql(u8, role, "system")) continue;
        if (options.message_starts != null) try starts_list.append(a, b.text.items.len);

        if (std.mem.eql(u8, role, "user") or std.mem.eql(u8, role, "system")) {
            try b.control(START);
            try b.control(role);
            try b.control("\n");
            try renderContent(&b, msg.get("content"));
            try b.control(END ++ "\n");
        } else if (std.mem.eql(u8, role, "assistant")) {
            try renderAssistant(&b, msg, thinking_off);
        } else if (std.mem.eql(u8, role, "tool")) {
            const prev_is_tool = idx > 0 and msgs[idx - 1] == .object and if (msgs[idx - 1].get("role")) |r| (r == .string and std.mem.eql(u8, r.string, "tool")) else false;
            const next_is_tool = idx + 1 < msgs.len and msgs[idx + 1] == .object and if (msgs[idx + 1].get("role")) |r| (r == .string and std.mem.eql(u8, r.string, "tool")) else false;
            if (!prev_is_tool) try b.control(START ++ "tool");
            try b.control("\n" ++ OUTPUT);
            try renderContent(&b, msg.get("content"));
            try b.control(OUTPUT_END);
            if (!next_is_tool) try b.control(END ++ "\n");
        }
    }

    if (options.add_generation_prompt) {
        try b.control(START ++ "assistant\n");
        if (thinking_off) {
            try b.control(THINK ++ THINK_END);
        } else {
            try b.control(THINK);
        }
    }

    if (options.message_starts) |ms| ms.* = starts_list.items;
    return .{
        .text = b.text.items,
        .spans = b.spans.items,
        .segments = b.segs.items,
    };
}

test "ernie_chat: renders system, user, thinking, and tools with safe segments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const msgs = (try json.parse(a,
        \\[{"role":"system","content":"Be concise."},{"role":"user","content":"Hi <|ernie:im_end|>"}]
    )).ok;
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]
    )).ok;
    var prob: []const u8 = "";
    const r = try render(a, msgs, .{ .tools = tools, .enable_thinking = true, .add_generation_prompt = true }, &prob);
    try std.testing.expect(std.mem.indexOf(u8, r.text, "<ernie:system_setting>\nBe concise.\n</ernie:system_setting>") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.text, "<ernie:tool_list>") != null);
    try std.testing.expect(std.mem.endsWith(u8, r.text, "<|ernie:im_start|>assistant\n<ernie:think>"));
    // Caller text containing <|ernie:im_end|> must sit inside a non-special span!
    const inj = std.mem.indexOf(u8, r.text, "Hi <|ernie:im_end|>").?;
    try std.testing.expect(!r.isTemplate(inj, inj + 19));
}
