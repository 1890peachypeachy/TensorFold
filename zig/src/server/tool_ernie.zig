//! ERNIE 5.5 tool calls (`<ernie:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></ernie:tool_call>`)
//! rewritten as DeepSeek-V4's DSML so the server's shared parser and streamer handle multi-arg, parallel, and typed args.
const std = @import("std");
const json = @import("json");
const tool_params = @import("tool_params.zig");
const tool_kinds = @import("tool_kinds.zig");
const reply_text = @import("reply_text.zig");

const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const open = "<ernie:tool_call>";
pub const close = "</ernie:tool_call>";
const dsml_open = reply_text.calls[2][0];
const dsml_close = reply_text.calls[2][1];

/// Rewrite each complete, well-formed `<ernie:tool_call>...</ernie:tool_call>` block to DSML.
pub fn dsml(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error![]const u8 {
    if (tools.len == 0 or std.mem.indexOf(u8, text, open) == null) return text;
    const schemas = try tool_params.schemas(a, tools);
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, open)) |s| {
        const e = std.mem.indexOfPos(u8, text, s + open.len, close) orelse break;
        try out.appendSlice(a, text[pos..s]);
        if (try block(a, text[s + open.len .. e], &schemas)) |b|
            try out.appendSlice(a, b)
        else
            try out.appendSlice(a, text[s .. e + close.len]);
        pos = e + close.len;
    }
    try out.appendSlice(a, text[pos..]);
    return out.items;
}

/// The close token's byte offset after complete ERNIE call blocks before it have been rewritten.
pub fn closeAfterRewrite(a: Allocator, raw: []const u8, tools: []const Value, where: reply_text.Close) Allocator.Error!reply_text.Close {
    return switch (where) {
        .token => |at| if (at <= raw.len and std.mem.indexOf(u8, raw[0..at], open) != null)
            .{ .token = (try dsml(a, raw[0..at], tools)).len }
        else
            where,
        else => where,
    };
}

fn safeName(text: []const u8) bool {
    return std.mem.indexOfScalar(u8, text, '"') == null and std.mem.indexOf(u8, text, "\u{ff5c}DSML\u{ff5c}") == null;
}

fn isNullWord(s: []const u8) bool {
    return std.ascii.eqlIgnoreCase(s, "null") or std.ascii.eqlIgnoreCase(s, "none") or std.ascii.eqlIgnoreCase(s, "nil");
}

fn boolWord(s: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(s, "yes") or std.ascii.eqlIgnoreCase(s, "on") or std.ascii.eqlIgnoreCase(s, "true")) return true;
    if (std.ascii.eqlIgnoreCase(s, "no") or std.ascii.eqlIgnoreCase(s, "off") or std.ascii.eqlIgnoreCase(s, "false")) return false;
    return null;
}

fn schemaAllowsBool(a: Allocator, schema: tool_params.Schema) Allocator.Error!bool {
    const resolved = (try tool_kinds.resolved(a, schema.value, schema.root)) orelse return false;
    if (resolved != .object) return false;
    if (resolved.get("type")) |t| {
        if (t == .string and std.mem.eql(u8, t.string, "boolean")) return true;
        if (t == .array) for (t.array) |x| if (x == .string and std.mem.eql(u8, x.string, "boolean")) return true;
    }
    for ([_][]const u8{ "anyOf", "oneOf" }) |comb| {
        if (resolved.get(comb)) |arr| if (arr == .array) for (arr.array) |branch| {
            if (try schemaAllowsBool(a, .{ .value = branch, .root = schema.root })) return true;
        };
    }
    return false;
}

fn schemaIsStringOnly(a: Allocator, schema: tool_params.Schema) Allocator.Error!bool {
    const resolved = (try tool_kinds.resolved(a, schema.value, schema.root)) orelse return false;
    if (resolved != .object) return false;
    if (resolved.get("type")) |t| {
        return t == .string and std.mem.eql(u8, t.string, "string");
    }
    return false;
}

fn decodeParam(a: Allocator, raw: []const u8, schema: tool_params.Schema) Allocator.Error!Value {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (isNullWord(trimmed) and !(try schemaIsStringOnly(a, schema))) {
        return .null;
    }
    if (try schemaAllowsBool(a, schema)) {
        if (boolWord(trimmed)) |bv| return .{ .bool = bv };
    }
    return tool_params.decode(a, raw, schema, false);
}

fn block(a: Allocator, body: []const u8, schemas: *const tool_params.Schemas) Allocator.Error!?[]const u8 {
    const invoke_open = "<invoke name=\"";
    const invoke_end = "</invoke>";
    const param_open = "<parameter name=\"";
    const param_end = "</parameter>";

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, dsml_open);
    var p: usize = 0;
    var calls: usize = 0;
    while (true) {
        const rest = std.mem.trimStart(u8, body[p..], " \t\r\n");
        p = body.len - rest.len;
        if (p == body.len) break;
        if (!std.mem.startsWith(u8, rest, invoke_open)) return null;
        const n0 = p + invoke_open.len;
        const q = std.mem.indexOfPos(u8, body, n0, "\">") orelse return null;
        const name = body[n0..q];
        if (name.len == 0 or !safeName(name)) return null;
        try out.print(a, "\n<\u{ff5c}DSML\u{ff5c}invoke name=\"{s}\">", .{name});
        var k = q + 2;
        while (true) {
            const r = std.mem.trimStart(u8, body[k..], " \t\r\n");
            k = body.len - r.len;
            if (std.mem.startsWith(u8, body[k..], invoke_end)) break;
            if (!std.mem.startsWith(u8, body[k..], param_open)) return null;
            const k0 = k + param_open.len;
            const kq = std.mem.indexOfPos(u8, body, k0, "\">") orelse return null;
            const pname = body[k0..kq];
            if (pname.len == 0 or !safeName(pname)) return null;
            const v0 = kq + 2;
            const ve = std.mem.indexOfPos(u8, body, v0, param_end) orelse return null;
            const raw_val = body[v0..ve];
            const schema = try tool_params.schemaOf(schemas, name, pname, a);
            const val = try decodeParam(a, raw_val, schema);
            const spelled = json.stringify(a, val, .{ .ascii = true }) catch return error.OutOfMemory;
            try out.print(a, "<\u{ff5c}DSML\u{ff5c}parameter name=\"{s}\" string=\"false\">{s}</\u{ff5c}DSML\u{ff5c}parameter>", .{ pname, spelled });
            k = ve + param_end.len;
        }
        try out.appendSlice(a, "</\u{ff5c}DSML\u{ff5c}invoke>");
        p = k + invoke_end.len;
        calls += 1;
    }
    if (calls == 0) return null;
    try out.appendSlice(a, "\n" ++ dsml_close);
    return out.items;
}

test "tool_ernie: multi-arg, parallel invokes, null/bool coercion, and streaming holdback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"q":{"type":"string"},"limit":{"type":"integer"},"exact":{"type":"boolean"},"filter":{"anyOf":[{"type":"string"},{"type":"null"}]}}}}},{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}]
    )).ok;
    const text = "Checking." ++ open ++ "\n<invoke name=\"search\"><parameter name=\"q\">hello</parameter><parameter name=\"limit\">5</parameter><parameter name=\"exact\">yes</parameter><parameter name=\"filter\">none</parameter></invoke>\n<invoke name=\"get_time\"><parameter name=\"tz\">UTC</parameter></invoke>\n" ++ close;
    const got = try dsml(a, text, tools.array);
    const parsed = try @import("tool_parse.zig").parse(a, got, tools.array, null);
    try std.testing.expectEqualStrings("Checking.", parsed.content);
    try std.testing.expectEqual(@as(usize, 2), parsed.calls.?.len);
    const args0 = (try json.parse(a, parsed.calls.?[0].get("function").?.get("arguments").?.string)).ok;
    try std.testing.expectEqualStrings("hello", args0.get("q").?.string);
    try std.testing.expectEqualStrings("5", args0.get("limit").?.int);
    try std.testing.expectEqual(true, args0.get("exact").?.bool);
    try std.testing.expect(args0.get("filter").? == .null);
    const args1 = (try json.parse(a, parsed.calls.?[1].get("function").?.get("arguments").?.string)).ok;
    try std.testing.expectEqualStrings("UTC", args1.get("tz").?.string);
    // Partial streaming block stays untouched until </ernie:tool_call> arrives
    const partial = "x" ++ open ++ "\n<invoke name=\"search\"><parameter name=\"q\">hel";
    try std.testing.expectEqualStrings(partial, try dsml(a, partial, tools.array));
}
