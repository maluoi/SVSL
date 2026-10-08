#!/usr/bin/env python3
# unroll_sweep.py - the unroll policy's evidence on a real GPU (docs/PLAN_optimizer_llvm.md,
# Phase 3 item 4). Generates encoder-shaped compute shaders - `T px[N]` filled from a
# texture, a min/max pass, a project/quantize/error loop over the array (optionally inside
# a rolled candidate loop), a write-back pass - compiles each with every given compiler,
# and measures them with svsl_isa: driver statistics (scratch, registers, instructions)
# and dispatch time.
#
#   unroll_sweep.py <work_dir> --isa <svsl_isa> --svslc NAME=PATH [--svslc NAME=PATH ...]
#                   [--loop NAME=PATH] [--skshaderc PATH] [--adb] [--grid small|full]
#
#   --svslc NAME=PATH  a compiler to compare. Research builds that force the policy:
#                      cmake -DCMAKE_C_FLAGS=-DUNROLL_MIN_ELEMENTS=0 (always unroll) or
#                      -DUNROLL_MIN_ELEMENTS=1000000 (never) - see src/ir/passes/unroll.c
#   --loop NAME=PATH   also compile with every [unroll] spelled [loop] (no unroll hint)
#   --skshaderc PATH   glslang + spirv-opt through skshaderc (needs spirv-as on PATH)
#   --adb              run on the attached Android device (svsl_isa built for Android,
#                      see CLAUDE.md); otherwise on this machine's GPU
#
# Prints one row per shader: per compiler, dispatch ms / scratch bytes / registers /
# instructions (each driver names its statistics; svsl_isa normalizes the common ones).

import argparse, itertools, os, re, subprocess, sys

SHADER = r'''Texture2D<float4>         source_tex : register(t0);
RWStructuredBuffer<uint4> results    : register(u1);
uint blocks_x;
uint blocks_y;

#ifndef UNROLL_HINT
#define UNROLL_HINT [unroll]
#endif
#define N     %(N)d
#define CANDS %(C)d
#define T     %(T)s
#define DOT(a, b) %(DOT)s

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= blocks_x || id.y >= blocks_y) return;
	int2 base = int2(id.xy * 8);
	T px[N];
	UNROLL_HINT for (int k = 0; k < N; k++) {
		float4 c = source_tex.Load(int3(base + int2(k & 7, k >> 3), 0));
		px[k] = %(LOAD)s;
	}
	T lo = px[0], hi = px[0];
	UNROLL_HINT for (int k1 = 1; k1 < N; k1++) { lo = min(lo, px[k1]); hi = max(hi, px[k1]); }
	float best = 1e30; uint best_c = 0; uint best_bits = 0;
#if CANDS > 1
	[loop] for (uint c = 0; c < CANDS; c++) {
#else
	{ uint c = 0;
#endif
		float f  = c * 0.04;
		T     a  = lerp(lo, hi, f);
		T     d  = lerp(hi, lo, f) - a;
		float dd = DOT(d, d) + 1e-6;
		float err = 0; uint bits = 0;
		UNROLL_HINT for (int k2 = 0; k2 < N; k2++) {
			float t = saturate(DOT(px[k2] - a, d) / dd);
			uint  q = (uint)(t * 7 + 0.5);
			T     e = px[k2] - (a + d * (q / 7.0));
			err += DOT(e, e);
			bits = bits * 3 + q;%(HEAVY)s
		}
		if (err < best) { best = err; best_c = c; best_bits = bits; }
	}
	UNROLL_HINT for (int k3 = 0; k3 < N; k3++) px[k3] = lerp(px[k3], lo, 0.25);
	float sum = 0;
	UNROLL_HINT for (int k4 = 0; k4 < N; k4++) sum += DOT(px[k4], px[k4]);
	results[id.y * blocks_x + id.x] = uint4(asuint(best), best_c, best_bits, asuint(sum));
}
'''
HEAVY = r'''
			T w = e * e * (1 + t) + px[k2] * 0.125;
			err += DOT(w, w) * 0.25;
			bits ^= (uint)(DOT(w, px[k2]) * 255.0) << (k2 & 7);'''

GRIDS = {
	'small': dict(n=[16, 32, 64], t=['float', 'float4'], body=['heavy'],          cands=[1, 16]),
	'full':  dict(n=[8, 16, 32, 64], t=['float', 'float4'], body=['light', 'heavy'], cands=[1, 16]),
}
BLOCKS   = 256 # blocks per side: one thread per 8x8 texel block of a 2048x2048 texture
DEVICE   = '/data/local/tmp/svsl_sweep'

def sh(cmd, **kw):
	return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)

def generate(src_dir, grid):
	g, names = GRIDS[grid], []
	for n, t, body, c in itertools.product(g['n'], g['t'], g['body'], g['cands']):
		name = 'n%d_%s_%s_c%d' % (n, 'f4' if t == 'float4' else 'f1', body, c)
		text = SHADER % dict(N=n, C=c, T=t, DOT='dot(a, b)' if t == 'float4' else '((a) * (b))',
		                     LOAD='c' if t == 'float4' else 'dot(c.rgb, float3(0.299, 0.587, 0.114))',
		                     HEAVY=HEAVY if body == 'heavy' else '')
		open(os.path.join(src_dir, name + '.hlsl'), 'w').write(text)
		names.append(name)
	return names

def compile_all(args, names, src, spv):
	variants = [(label, path, []) for label, path in args.svslc]
	variants += [(label, path, ['-D', 'UNROLL_HINT=[loop]']) for label, path in args.loop]
	for name in names:
		for label, path, extra in variants:
			out = os.path.join(spv, '%s.%s.spv' % (name, label))
			r = sh([path, '-f', '-spv', '-O1', *extra, '-o', out, os.path.join(src, name + '.hlsl')])
			if r.returncode: sys.exit('%s failed on %s:\n%s' % (label, name, r.stderr))
		if args.skshaderc:
			r = sh([args.skshaderc, '-raw', '-t', 's', '-cs', 'cs', '-o', spv, os.path.join(src, name + '.hlsl')], cwd=spv)
			asm = os.path.join(spv, name + '.hlsl.compute.spvasm')
			r = sh(['spirv-as', '--target-env', 'vulkan1.1', asm, '-o', os.path.join(spv, name + '.glslang.spv')])
			if r.returncode: sys.exit('glslang path failed on %s:\n%s' % (name, r.stderr))
	return [label for label, _, _ in variants] + (['glslang'] if args.skshaderc else [])

def measure(args, spv, file):
	run = ['run', file, '-groups', str(BLOCKS // 8), str(BLOCKS // 8), '1', '-tex', '2048', '2048',
	       '-u32', '0=%d' % BLOCKS, '-u32', '1=%d' % BLOCKS, '-iters', '40', '-warm', '400']
	if args.adb:
		dev = '%s/%s' % (DEVICE, os.path.basename(file))
		sh(['adb', 'push', file, dev])
		out = sh(['adb', 'shell', 'cd %s && timeout 60 ./svsl_isa %s; timeout 60 ./svsl_isa %s' %
		          (DEVICE, dev, ' '.join([run[0], dev] + run[2:]))]).stdout
	else:
		out = sh([args.isa, file]).stdout + sh([args.isa] + run).stdout
	stat = lambda key: (re.search(r'\b%s=(\d+)' % key, out) or [None, '-'])[1]
	ms   = re.search(r'median ([\d.]+) ms', out)
	return '%7s %5s %3s %6s' % (ms.group(1) if ms else 'fail', stat('scratch'), stat('regs') if stat('regs') != '-' else stat('vgpr'), stat('insts'))

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('work_dir')
	ap.add_argument('--isa', required=True)
	ap.add_argument('--svslc', action='append', default=[], type=lambda s: s.split('=', 1))
	ap.add_argument('--loop', action='append', default=[], type=lambda s: s.split('=', 1))
	ap.add_argument('--skshaderc')
	ap.add_argument('--adb', action='store_true')
	ap.add_argument('--grid', default='small', choices=GRIDS)
	args = ap.parse_args()
	if not args.svslc: sys.exit('give at least one --svslc NAME=PATH')
	src, spv = os.path.join(args.work_dir, 'src'), os.path.join(args.work_dir, 'spv')
	os.makedirs(src, exist_ok=True); os.makedirs(spv, exist_ok=True)

	names  = generate(src, args.grid)
	labels = compile_all(args, names, src, spv)
	if args.adb:
		sh(['adb', 'shell', 'mkdir -p %s' % DEVICE])
		sh(['adb', 'push', args.isa, DEVICE + '/svsl_isa'])
		sh(['adb', 'shell', 'chmod +x %s/svsl_isa' % DEVICE])
	print('%-20s | %s' % ('shader', ' | '.join('%-24s' % l for l in labels)))
	print('%-20s | %s' % ('', ' | '.join(['     ms scrat reg  insts'] * len(labels))))
	for name in names:
		row = [measure(args, spv, os.path.join(spv, '%s.%s.spv' % (name, l))) for l in labels]
		print('%-20s | %s' % (name, ' | '.join(row)), flush=True)

main()
