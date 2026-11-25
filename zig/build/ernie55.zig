//! ERNIE 5.5 Flash Preview CUDA family build wiring: module, standalone verification tool, and host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/ernie55/ernie55.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    return m;
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const mod = module(b, target, optimize, cuda, core, lanes);
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/ernie55/cuda_dump_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    m.addImport("cuda", cuda);
    m.addImport("ernie55", mod);
    const exe = b.addExecutable(.{ .name = "tf-ernie55-cuda", .root_module = m });
    b.step("tf-ernie55-cuda", "ERNIE 5.5 Flash Preview CUDA forward parity and speed tool").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

pub fn hostTests(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, lanes) }));
    b.step("test-ernie55-host", "ERNIE 5.5 host tests: config, TP4 sharding, RoPE, E4M3 decode (no GPU)").dependOn(&run.step);
    all.dependOn(&run.step);
}
