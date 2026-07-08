#include "intrinsics.h"

#include "../../vendor/spirv.h"
#include "../../vendor/GLSL.std.450.h"

#include <string.h>

// table rows initialize the fields they need; zero-fill for the rest is the point
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#define P1(a)          { a, 0 }
#define P2(a, b)       { a, b, 0 }
#define P3(a, b, c)    { a, b, c, 0 }
#define P4(a, b, c, d) { a, b, c, d, 0 }

// back-end lowering per row (see svsl_emit_): EXT/EXT3 = GLSLstd450 ext inst,
// CORE = plain SpvOp, EM = a custom emit category
#define EXT(f)      .emit = svsl_emit_ext450, .op = { f }
#define EXT3(f,s,u) .emit = svsl_emit_ext450, .op = { f, s, u }
#define CORE(o)     .emit = svsl_emit_core,   .op = { o }
#define EM(e)       .emit = e

#define GENF   svsl_iparam_genf
#define GEN    svsl_iparam_gen
#define GENI   svsl_iparam_geni
#define GENB   svsl_iparam_genb
#define SAME   svsl_iparam_same
#define SCAL   svsl_iparam_scalar
#define UINT   svsl_iparam_uint
#define VEC3F  svsl_iparam_vec3f
#define ANY32  svsl_iparam_any32

static const svsl_intrinsic_t intrinsic_table[] = {
	// --- trig / exponential -------------------------------------------------------
	{ "sin",   P1(GENF), svsl_ires_gen, EXT(GLSLstd450Sin) }, { "cos",   P1(GENF), svsl_ires_gen, EXT(GLSLstd450Cos) },
	{ "tan",   P1(GENF), svsl_ires_gen, EXT(GLSLstd450Tan) }, { "asin",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Asin) },
	{ "acos",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Acos) }, { "atan",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Atan) },
	{ "sinh",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Sinh) }, { "cosh",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Cosh) },
	{ "tanh",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Tanh) }, { "atan2", P2(GENF, SAME), svsl_ires_gen, EXT(GLSLstd450Atan2) },
	{ "pow",   P2(GENF, SAME), svsl_ires_gen, EXT(GLSLstd450Pow) },
	{ "exp",   P1(GENF), svsl_ires_gen, EXT(GLSLstd450Exp) }, { "exp2",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Exp2) },
	{ "log",   P1(GENF), svsl_ires_gen, EXT(GLSLstd450Log) }, { "log2",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Log2) },
	{ "log10", P1(GENF), svsl_ires_gen, EM(svsl_emit_log10) },
	{ "sqrt",  P1(GENF), svsl_ires_gen, EXT(GLSLstd450Sqrt) }, { "rsqrt", P1(GENF), svsl_ires_gen, EXT(GLSLstd450InverseSqrt) },
	{ "rcp",   P1(GENF), svsl_ires_gen, EM(svsl_emit_rcp) },
	{ "degrees", P1(GENF), svsl_ires_gen, EXT(GLSLstd450Degrees) }, { "radians", P1(GENF), svsl_ires_gen, EXT(GLSLstd450Radians) },
	{ "ldexp",   P2(GENF, SAME), svsl_ires_gen, EM(svsl_emit_ldexp) },
	{ "sincos",  {0}, svsl_ires_special, .tag = svsl_intr_sincos }, // (x, out s, out c)
	{ "modf",    {0}, svsl_ires_special, .tag = svsl_intr_modf }, // (x, out ip) → frac
	{ "frexp",   {0}, svsl_ires_special, .tag = svsl_intr_frexp }, // (x, out exp) → mantissa
	{ "frexp_mant", P1(GENF), svsl_ires_gen, EM(svsl_emit_frexp_mant) },  // internal halves of frexp
	{ "frexp_exp",  P1(GENF), svsl_ires_float_shape, EM(svsl_emit_frexp_exp) },
	{ "lit",   {0}, svsl_ires_special, .tag = svsl_intr_legacy }, // legacy D3D9, rejected with a message
	{ "dst",   {0}, svsl_ires_special, .tag = svsl_intr_legacy },
	{ "msad4", {0}, svsl_ires_special, .tag = svsl_intr_legacy },

	// --- common math --------------------------------------------------------------
	{ "abs",      P1(GEN),  svsl_ires_gen, EXT3(GLSLstd450FAbs, GLSLstd450SAbs, 0) },
	{ "sign",     P1(GEN),  svsl_ires_gen, EXT3(GLSLstd450FSign, GLSLstd450SSign, 0) },
	{ "floor",    P1(GENF), svsl_ires_gen, EXT(GLSLstd450Floor) }, { "ceil",     P1(GENF), svsl_ires_gen, EXT(GLSLstd450Ceil) },
	{ "trunc",    P1(GENF), svsl_ires_gen, EXT(GLSLstd450Trunc) }, { "round",    P1(GENF), svsl_ires_gen, EXT(GLSLstd450Round) },
	{ "frac",     P1(GENF), svsl_ires_gen, EXT(GLSLstd450Fract) }, { "fmod",     P2(GENF, SAME), svsl_ires_gen, CORE(SpvOpFRem) },
	{ "min",      P2(GEN, SAME),  svsl_ires_gen, EXT3(GLSLstd450FMin, GLSLstd450SMin, GLSLstd450UMin) },
	{ "max",      P2(GEN, SAME),  svsl_ires_gen, EXT3(GLSLstd450FMax, GLSLstd450SMax, GLSLstd450UMax) },
	{ "clamp",    P3(GEN, SAME, SAME),   svsl_ires_gen, EXT3(GLSLstd450FClamp, GLSLstd450SClamp, GLSLstd450UClamp) },
	{ "saturate", P1(GENF), svsl_ires_gen, EM(svsl_emit_saturate) },
	{ "lerp",     P3(GENF, SAME, SAME),  svsl_ires_gen, EXT(GLSLstd450FMix) },
	{ "step",     P2(GENF, SAME), svsl_ires_gen, EXT(GLSLstd450Step) },
	{ "smoothstep", P3(GENF, SAME, SAME), svsl_ires_gen, EXT(GLSLstd450SmoothStep) },
	{ "fma",      P3(GENF, SAME, SAME),  svsl_ires_gen, EXT(GLSLstd450Fma) },
	{ "mad",      P3(GENF, SAME, SAME),  svsl_ires_gen, EXT(GLSLstd450Fma) },

	// --- vector -------------------------------------------------------------------
	{ "length",      P1(GENF),        svsl_ires_scalar, EXT(GLSLstd450Length) },
	{ "distance",    P2(GENF, SAME),  svsl_ires_scalar, EXT(GLSLstd450Distance) },
	{ "normalize",   P1(GENF),        svsl_ires_gen,    EXT(GLSLstd450Normalize) },
	{ "dot",         P2(GENF, SAME),  svsl_ires_scalar, CORE(SpvOpDot) },
	{ "cross",       P2(VEC3F, SAME), svsl_ires_gen,    EXT(GLSLstd450Cross) },
	{ "reflect",     P2(GENF, SAME),  svsl_ires_gen,    EXT(GLSLstd450Reflect) },
	{ "refract",     P3(GENF, SAME, SCAL), svsl_ires_gen, EXT(GLSLstd450Refract) },
	{ "faceforward", P3(GENF, SAME, SAME), svsl_ires_gen, EXT(GLSLstd450FaceForward) },

	// --- matrix (shape-special) -----------------------------------------------------
	{ "mul",         {0}, svsl_ires_special, .tag = svsl_intr_mul },
	{ "transpose",   {0}, svsl_ires_special, .tag = svsl_intr_transpose,   CORE(SpvOpTranspose) },
	{ "determinant", {0}, svsl_ires_special, .tag = svsl_intr_determinant, EXT(GLSLstd450Determinant) },
	{ "inverse",     {0}, svsl_ires_special, .tag = svsl_intr_inverse,     EXT(GLSLstd450MatrixInverse) },

	// --- logic / classification ------------------------------------------------------
	{ "any",    P1(GENB), svsl_ires_bool, EM(svsl_emit_any) }, { "all",    P1(GENB), svsl_ires_bool, EM(svsl_emit_all) },
	{ "select", {0},      svsl_ires_special, .tag = svsl_intr_select },
	{ "isnan",  P1(GENF), svsl_ires_bool_shape, CORE(SpvOpIsNan) },
	{ "isinf",  P1(GENF), svsl_ires_bool_shape, CORE(SpvOpIsInf) },

	// --- bits ---------------------------------------------------------------------
	{ "countbits",    P1(GENI), svsl_ires_uint_shape, CORE(SpvOpBitCount) },
	{ "reversebits",  P1(GENI), svsl_ires_gen,        CORE(SpvOpBitReverse) },
	{ "firstbithigh", P1(GENI), svsl_ires_int_shape,  EXT3(GLSLstd450FindSMsb, GLSLstd450FindSMsb, GLSLstd450FindUMsb) },
	{ "firstbitlow",  P1(GENI), svsl_ires_int_shape,  EXT(GLSLstd450FindILsb) },
	// (value, offset, bits) → the [offset, offset+bits) field, right-justified;
	// signed value sign-extends. insert replaces that field of base with value's
	// low bits. offset/bits are uint scalars; map to OpBitFieldU/SExtract / Insert.
	{ "bitfield_extract", P3(GENI, UINT, UINT),       svsl_ires_gen, EM(svsl_emit_bitfield_extract) },
	{ "bitfield_insert",  P4(GENI, SAME, UINT, UINT), svsl_ires_gen, EM(svsl_emit_bitfield_insert) },

	// --- reinterpret / pack ----------------------------------------------------------
	{ "asfloat",  P1(ANY32), svsl_ires_float_shape, EM(svsl_emit_bitcast) },
	{ "asuint",   P1(ANY32), svsl_ires_uint_shape,  EM(svsl_emit_bitcast) },
	{ "asint",    P1(ANY32), svsl_ires_int_shape,   EM(svsl_emit_bitcast) },
	{ "f16tof32", P1(GENI),  svsl_ires_float_shape, EM(svsl_emit_f16tof32) },
	{ "f32tof16", P1(GENF),  svsl_ires_uint_shape,  EM(svsl_emit_f32tof16) },
	{ "pack_unorm4x8",   {svsl_iparam_genf, 0}, svsl_ires_special, .tag = svsl_intr_pack4, EXT(GLSLstd450PackUnorm4x8) },
	{ "unpack_unorm4x8", P1(UINT), svsl_ires_float4, EXT(GLSLstd450UnpackUnorm4x8) },
	{ "pack_snorm4x8",   {svsl_iparam_genf, 0}, svsl_ires_special, .tag = svsl_intr_pack4, EXT(GLSLstd450PackSnorm4x8) },
	{ "unpack_snorm4x8", P1(UINT), svsl_ires_float4, EXT(GLSLstd450UnpackSnorm4x8) },
	{ "pack_half2x16",   {svsl_iparam_genf, 0}, svsl_ires_special, .tag = svsl_intr_pack2, EXT(GLSLstd450PackHalf2x16) },
	{ "unpack_half2x16", P1(UINT), svsl_ires_float2, EXT(GLSLstd450UnpackHalf2x16) },

	// --- fragment ------------------------------------------------------------------
	{ "clip",       P1(GENF), svsl_ires_void, EM(svsl_emit_clip) },
	{ "ddx",        P1(GENF), svsl_ires_gen, CORE(SpvOpDPdx) }, { "ddy",        P1(GENF), svsl_ires_gen, CORE(SpvOpDPdy) },
	{ "ddx_coarse", P1(GENF), svsl_ires_gen, CORE(SpvOpDPdxCoarse) }, { "ddy_coarse", P1(GENF), svsl_ires_gen, CORE(SpvOpDPdyCoarse) },
	{ "ddx_fine",   P1(GENF), svsl_ires_gen, CORE(SpvOpDPdxFine) }, { "ddy_fine",   P1(GENF), svsl_ires_gen, CORE(SpvOpDPdyFine) },
	{ "fwidth",     P1(GENF), svsl_ires_gen, CORE(SpvOpFwidth) },
	{ "is_helper_invocation", {0}, svsl_ires_bool, EM(svsl_emit_is_helper) },
	{ "tile_depth",   {0}, svsl_ires_float_shape, EM(svsl_emit_tile_depth) }, // SPV_EXT_shader_tile_image reads
	{ "tile_stencil", {0}, svsl_ires_uint,        EM(svsl_emit_tile_stencil) },

	// --- barriers (native + HLSL aliases) ----------------------------------------------
	{ "workgroup_barrier",          {0}, svsl_ires_void, EM(svsl_emit_barrier_wg) },
	{ "workgroup_memory_barrier",   {0}, svsl_ires_void, EM(svsl_emit_barrier_wg_mem) },
	{ "device_memory_barrier",      {0}, svsl_ires_void, EM(svsl_emit_barrier_device) },
	{ "all_memory_barrier",         {0}, svsl_ires_void, EM(svsl_emit_barrier_all) },
	{ "device_memory_barrier_sync", {0}, svsl_ires_void, EM(svsl_emit_barrier_device_sync) },
	{ "all_memory_barrier_sync",    {0}, svsl_ires_void, EM(svsl_emit_barrier_all_sync) },
	{ "GroupMemoryBarrierWithGroupSync", {0}, svsl_ires_void, "workgroup_barrier",          EM(svsl_emit_barrier_wg) },
	{ "GroupMemoryBarrier",              {0}, svsl_ires_void, "workgroup_memory_barrier",   EM(svsl_emit_barrier_wg_mem) },
	{ "DeviceMemoryBarrier",             {0}, svsl_ires_void, "device_memory_barrier",      EM(svsl_emit_barrier_device) },
	{ "AllMemoryBarrier",                {0}, svsl_ires_void, "all_memory_barrier",         EM(svsl_emit_barrier_all) },
	{ "DeviceMemoryBarrierWithGroupSync",{0}, svsl_ires_void, "device_memory_barrier_sync", EM(svsl_emit_barrier_device_sync) },
	{ "AllMemoryBarrierWithGroupSync",   {0}, svsl_ires_void, "all_memory_barrier_sync",    EM(svsl_emit_barrier_all_sync) },

	// --- atomics (lvalue checks are shape-special) ---------------------------------------
	{ "atomic_add",              {0}, svsl_ires_special, .tag = svsl_intr_atomic_add },
	{ "atomic_sub",              {0}, svsl_ires_special, .tag = svsl_intr_atomic_sub },
	{ "atomic_min",              {0}, svsl_ires_special, .tag = svsl_intr_atomic_min },
	{ "atomic_max",              {0}, svsl_ires_special, .tag = svsl_intr_atomic_max },
	{ "atomic_and",              {0}, svsl_ires_special, .tag = svsl_intr_atomic_and },
	{ "atomic_or",               {0}, svsl_ires_special, .tag = svsl_intr_atomic_or },
	{ "atomic_xor",              {0}, svsl_ires_special, .tag = svsl_intr_atomic_xor },
	{ "atomic_exchange",         {0}, svsl_ires_special, .tag = svsl_intr_atomic_exchange },
	{ "atomic_compare_exchange", {0}, svsl_ires_special, .tag = svsl_intr_atomic_cmpxchg },
	{ "InterlockedAdd",              {0}, svsl_ires_special, "atomic_add", .tag = svsl_intr_atomic_add },
	{ "InterlockedMin",              {0}, svsl_ires_special, "atomic_min", .tag = svsl_intr_atomic_min },
	{ "InterlockedMax",              {0}, svsl_ires_special, "atomic_max", .tag = svsl_intr_atomic_max },
	{ "InterlockedAnd",              {0}, svsl_ires_special, "atomic_and", .tag = svsl_intr_atomic_and },
	{ "InterlockedOr",               {0}, svsl_ires_special, "atomic_or", .tag = svsl_intr_atomic_or },
	{ "InterlockedXor",              {0}, svsl_ires_special, "atomic_xor", .tag = svsl_intr_atomic_xor },
	{ "InterlockedExchange",         {0}, svsl_ires_special, "atomic_exchange", .tag = svsl_intr_atomic_exchange },
	{ "InterlockedCompareExchange",  {0}, svsl_ires_special, "atomic_compare_exchange", .tag = svsl_intr_atomic_cmpxchg },
	{ "InterlockedCompareStore",     {0}, svsl_ires_special, "atomic_compare_exchange", .tag = svsl_intr_atomic_cmpxchg },

	// --- subgroup operations (all route to the subgroup sub-dispatcher) --------------
	{ "subgroup_elect",           {0},        svsl_ires_bool,  EM(svsl_emit_subgroup) },
	{ "subgroup_ballot",          {svsl_iparam_bool, 0}, svsl_ires_uint4, EM(svsl_emit_subgroup) },
	{ "subgroup_ballot_bit_count",           {svsl_iparam_bool, 0}, svsl_ires_uint, EM(svsl_emit_subgroup) },
	{ "subgroup_ballot_exclusive_bit_count", {svsl_iparam_bool, 0}, svsl_ires_uint, EM(svsl_emit_subgroup) },
	{ "subgroup_all",             {svsl_iparam_bool, 0}, svsl_ires_bool, EM(svsl_emit_subgroup) },
	{ "subgroup_any",             {svsl_iparam_bool, 0}, svsl_ires_bool, EM(svsl_emit_subgroup) },
	{ "subgroup_all_equal",       P1(GEN),   svsl_ires_bool, EM(svsl_emit_subgroup) },
	{ "subgroup_broadcast",       P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_broadcast_first", P1(GEN),   svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_add", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_mul", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_min", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_max", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_and", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_or", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_xor", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_add", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_add", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_mul", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_mul", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_min", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_min", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_max", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_max", P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_and", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_and", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_or",  P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_or",  P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_inclusive_xor", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) }, { "subgroup_exclusive_xor", P1(GENI), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_add", P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_mul", P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_min", P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_max", P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_and", P2(GENI, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_or",  P2(GENI, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_clustered_xor", P2(GENI, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_shuffle",      P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_shuffle_xor",  P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_shuffle_up",   P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "subgroup_shuffle_down", P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "quad_broadcast",        P2(GEN, UINT), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "quad_swap_horizontal",  P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "quad_swap_vertical",    P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },
	{ "quad_swap_diagonal",    P1(GEN), svsl_ires_gen, EM(svsl_emit_subgroup) },

	// --- HLSL Wave* aliases (emit follows the native subgroup op) -------------------------
	{ "WaveGetLaneCount",    {0}, svsl_ires_uint, "subgroup_size",    EM(svsl_emit_builtin_var) },
	{ "WaveGetLaneIndex",    {0}, svsl_ires_uint, "subgroup_lane_id", EM(svsl_emit_builtin_var) },
	{ "WaveIsFirstLane",     {0}, svsl_ires_bool, "subgroup_elect",   EM(svsl_emit_subgroup) },
	{ "WaveActiveAnyTrue",   {svsl_iparam_bool, 0}, svsl_ires_bool,  "subgroup_any",       EM(svsl_emit_subgroup) },
	{ "WaveActiveAllTrue",   {svsl_iparam_bool, 0}, svsl_ires_bool,  "subgroup_all",       EM(svsl_emit_subgroup) },
	{ "WaveActiveAllEqual",  P1(GEN),  svsl_ires_bool,  "subgroup_all_equal",   EM(svsl_emit_subgroup) },
	{ "WaveActiveBallot",    {svsl_iparam_bool, 0}, svsl_ires_uint4, "subgroup_ballot",    EM(svsl_emit_subgroup) },
	{ "WaveActiveSum",       P1(GEN),  svsl_ires_gen,   "subgroup_add",         EM(svsl_emit_subgroup) },
	{ "WaveActiveProduct",   P1(GEN),  svsl_ires_gen,   "subgroup_mul",         EM(svsl_emit_subgroup) },
	{ "WaveActiveMin",       P1(GEN),  svsl_ires_gen,   "subgroup_min",         EM(svsl_emit_subgroup) },
	{ "WaveActiveMax",       P1(GEN),  svsl_ires_gen,   "subgroup_max",         EM(svsl_emit_subgroup) },
	{ "WaveActiveBitAnd",    P1(GENI), svsl_ires_gen,   "subgroup_and",         EM(svsl_emit_subgroup) },
	{ "WaveActiveBitOr",     P1(GENI), svsl_ires_gen,   "subgroup_or",          EM(svsl_emit_subgroup) },
	{ "WaveActiveBitXor",    P1(GENI), svsl_ires_gen,   "subgroup_xor",         EM(svsl_emit_subgroup) },
	{ "WavePrefixSum",       P1(GEN),  svsl_ires_gen,   "subgroup_exclusive_add", EM(svsl_emit_subgroup) },
	{ "WavePrefixProduct",   P1(GEN),  svsl_ires_gen,   "subgroup_exclusive_mul", EM(svsl_emit_subgroup) },
	{ "WaveReadLaneAt",      P2(GEN, UINT), svsl_ires_gen, "subgroup_broadcast", EM(svsl_emit_subgroup) },
	{ "WaveReadLaneFirst",   P1(GEN),  svsl_ires_gen,   "subgroup_broadcast_first", EM(svsl_emit_subgroup) },
	{ "WaveActiveCountBits", {svsl_iparam_bool, 0}, svsl_ires_uint, "subgroup_ballot_bit_count", EM(svsl_emit_subgroup) },
	{ "WavePrefixCountBits", {svsl_iparam_bool, 0}, svsl_ires_uint, "subgroup_ballot_exclusive_bit_count", EM(svsl_emit_subgroup) },
	{ "QuadReadAcrossX",     P1(GEN),  svsl_ires_gen,   "quad_swap_horizontal", EM(svsl_emit_subgroup) },
	{ "QuadReadAcrossY",     P1(GEN),  svsl_ires_gen,   "quad_swap_vertical",   EM(svsl_emit_subgroup) },
	{ "QuadReadAcrossDiagonal", P1(GEN), svsl_ires_gen, "quad_swap_diagonal",   EM(svsl_emit_subgroup) },
	{ "QuadReadLaneAt",      P2(GEN, UINT), svsl_ires_gen, "quad_broadcast",    EM(svsl_emit_subgroup) },
};

int32_t svsl_intrinsic_find(svsl_str_t name) {
	for (int32_t i = 0; i < (int32_t)(sizeof(intrinsic_table) / sizeof(intrinsic_table[0])); i++)
		if (svsl_str_eq_cstr(name, intrinsic_table[i].name)) return i;
	return -1;
}

const svsl_intrinsic_t *svsl_intrinsic_get(int32_t index) {
	return &intrinsic_table[index];
}

const char *svsl_intrinsic_get_name(int32_t index) {
	return intrinsic_table[index].name;
}

// --- builtin read-only variables ----------------------------------------------------

static const char *builtin_vars[] = {
	"subgroup_size", "subgroup_lane_id", "subgroup_id", "num_subgroups",
};

int32_t svsl_builtin_var_find(svsl_str_t name) {
	for (int32_t i = 0; i < (int32_t)(sizeof(builtin_vars) / sizeof(builtin_vars[0])); i++)
		if (svsl_str_eq_cstr(name, builtin_vars[i])) return i;
	return -1;
}

// --- resource methods ------------------------------------------------------------------

typedef struct method_row_t {
	const char  *name;
	svsl_method_ method;
	int32_t      aux; // gather channel (0..3, -1 = all) or atomic op
} method_row_t;

static const method_row_t method_table[] = {
	{ "Sample",             svsl_method_sample,                -1 },
	{ "SampleLevel",        svsl_method_sample_level,          -1 },
	{ "SampleBias",         svsl_method_sample_bias,           -1 },
	{ "SampleGrad",         svsl_method_sample_grad,           -1 },
	{ "SampleCmp",          svsl_method_sample_cmp,            -1 },
	{ "SampleCmpLevelZero", svsl_method_sample_cmp_level_zero, -1 },
	{ "Gather",             svsl_method_gather,                -1 },
	{ "GatherRed",          svsl_method_gather,                 0 },
	{ "GatherGreen",        svsl_method_gather,                 1 },
	{ "GatherBlue",         svsl_method_gather,                 2 },
	{ "GatherAlpha",        svsl_method_gather,                 3 },
	{ "GatherCmp",          svsl_method_gather,                 4 },
	{ "Load",               svsl_method_load,                  -1 },
	{ "SubpassLoad",        svsl_method_load,                  -1 }, // DXC's SubpassInput spelling
	{ "Store",              svsl_method_store,                 -1 },
	{ "GetDimensions",      svsl_method_get_dimensions,        -1 },
	{ "CalculateLevelOfDetail",          svsl_method_query_lod,  0 },
	{ "CalculateLevelOfDetailUnclamped", svsl_method_query_lod,  1 },
	// aux = the emit atomic op index (svsl_intr_atomic_ order: add,sub,min,max,and,or,xor,exchange);
	// index 1 (sub) is skipped — images have no InterlockedSub — so the image-atomic path indexes
	// signed_ops[]/unsigned_ops[] identically to buffer atomics.
	{ "InterlockedAdd",     svsl_method_atomic,                 0 },
	{ "InterlockedMin",     svsl_method_atomic,                 2 },
	{ "InterlockedMax",     svsl_method_atomic,                 3 },
	{ "InterlockedAnd",     svsl_method_atomic,                 4 },
	{ "InterlockedOr",      svsl_method_atomic,                 5 },
	{ "InterlockedXor",     svsl_method_atomic,                 6 },
	{ "InterlockedExchange",svsl_method_atomic,                 7 },
};

int32_t svsl_method_find(svsl_str_t name, int32_t *out_channel_or_op) {
	for (int32_t i = 0; i < (int32_t)(sizeof(method_table) / sizeof(method_table[0])); i++) {
		if (svsl_str_eq_cstr(name, method_table[i].name)) {
			*out_channel_or_op = method_table[i].aux;
			return (int32_t)method_table[i].method;
		}
	}
	return -1;
}

int32_t svsl_atomic_order(svsl_str_t name) {
	static const struct { const char *name; int32_t order; } order_table[] = {
		{ "relaxed", svsl_mem_order_relaxed },
		{ "acquire", svsl_mem_order_acquire },
		{ "release", svsl_mem_order_release },
		{ "acq_rel", svsl_mem_order_acq_rel },
		{ "seq_cst", svsl_mem_order_seq_cst },
	};
	for (int32_t i = 0; i < (int32_t)(sizeof(order_table) / sizeof(order_table[0])); i++)
		if (svsl_str_eq_cstr(name, order_table[i].name)) return order_table[i].order;
	return -1;
}
