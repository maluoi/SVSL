// Intrinsic function and resource-method tables. Adding an intrinsic is a table
// row; only shape-special cases (mul, transpose, ...) have code, in sema/check.c.

#pragma once

#include "../util/str.h"

#include <stdint.h>

// generic parameter classes; 'G' is bound by the first generic argument
typedef enum svsl_iparam_ {
	svsl_iparam_end = 0,
	svsl_iparam_genf,    // float-family scalar/vector (half/float16/float32/float64) — binds G
	svsl_iparam_gen,     // any numeric scalar/vector — binds G
	svsl_iparam_geni,    // int/uint scalar/vector — binds G
	svsl_iparam_genb,    // bool scalar/vector (numeric converts) — binds G
	svsl_iparam_same,    // exactly G
	svsl_iparam_scalar,  // scalar of G's component type
	svsl_iparam_uint,    // uint32 scalar
	svsl_iparam_bool,    // bool scalar
	svsl_iparam_vec3f,   // float-family 3-vector — binds G
	svsl_iparam_any32,   // any 32-bit-component scalar/vector — binds G (asfloat & co)
} svsl_iparam_;

typedef enum svsl_ires_ {
	svsl_ires_gen = 0,    // G
	svsl_ires_scalar,     // component scalar of G
	svsl_ires_bool_shape, // bool with G's shape
	svsl_ires_bool,       // bool scalar
	svsl_ires_uint_shape, // uint with G's shape
	svsl_ires_int_shape,  // int with G's shape
	svsl_ires_float_shape,// float32 with G's shape
	svsl_ires_uint,       // uint scalar
	svsl_ires_uint4,
	svsl_ires_float2,
	svsl_ires_float4,
	svsl_ires_void,
	svsl_ires_special,    // resolved in code (mul, transpose, select, atomics, ...)
} svsl_ires_;

// semantic identity for intrinsics that any pipeline stage special-cases.
// Stages dispatch on the tag, never by parsing the name, so every layer
// agrees by construction. The atomic tags are ordered so they double as the
// IR/emit atomic op code.
typedef enum svsl_intr_tag_ {
	svsl_intr_none = 0,   // plain table intrinsic (emit maps the native name)
	svsl_intr_mul,
	svsl_intr_transpose,
	svsl_intr_determinant,
	svsl_intr_inverse,
	svsl_intr_select,
	svsl_intr_pack2,      // pack_half2x16: float2 → uint
	svsl_intr_pack4,      // pack_[us]norm4x8: float4 → uint
	svsl_intr_legacy,     // lit/dst/msad4: rejected with direction
	svsl_intr_sincos,
	svsl_intr_modf,
	svsl_intr_frexp,
	svsl_intr_atomic_add, // ... in IR atomic op-code order
	svsl_intr_atomic_sub,
	svsl_intr_atomic_min,
	svsl_intr_atomic_max,
	svsl_intr_atomic_and,
	svsl_intr_atomic_or,
	svsl_intr_atomic_xor,
	svsl_intr_atomic_exchange,
	svsl_intr_atomic_cmpxchg,
} svsl_intr_tag_;

#define SVSL_INTR_IS_ATOMIC(tag) ((tag) >= svsl_intr_atomic_add)
#define SVSL_INTR_ATOMIC_OP(tag) ((uint32_t)(tag) - svsl_intr_atomic_add)

// Optional memory-order name on a native atomic_* call (the trailing argument).
// Returns the order code (0 relaxed … 4 seq_cst) or -1 if the name is not one.
// emit maps the code to SPIR-V memory semantics; scope is inferred from storage.
typedef enum svsl_mem_order_ {
	svsl_mem_order_relaxed = 0,
	svsl_mem_order_acquire,
	svsl_mem_order_release,
	svsl_mem_order_acq_rel,
	svsl_mem_order_seq_cst,
} svsl_mem_order_;

int32_t svsl_atomic_order(svsl_str_t name);

// How the back end lowers an intrinsic. The IR stores the table index, so emit
// dispatches on this row field — the name is matched once, at find time, never
// re-parsed. `none` means the intrinsic is lowered before emit (mul, atomics,
// sincos…) or has no mapping (select) → emit errors, matching the old fall-through.
typedef enum svsl_emit_ {
	svsl_emit_none = 0,
	svsl_emit_ext450,    // GLSLstd450 ext inst; op[0..2] = float/signed/unsigned variant
	svsl_emit_core,      // core SpvOp in op[0] (derivative ops keyed by opcode range)
	svsl_emit_rcp, svsl_emit_ldexp, svsl_emit_log10, svsl_emit_saturate,
	svsl_emit_bitfield_extract, svsl_emit_bitfield_insert,
	svsl_emit_frexp_mant, svsl_emit_frexp_exp,
	svsl_emit_any, svsl_emit_all, svsl_emit_bitcast,        // asfloat/asuint/asint
	svsl_emit_f16tof32, svsl_emit_f32tof16,
	svsl_emit_clip, svsl_emit_tile_depth, svsl_emit_tile_stencil, svsl_emit_is_helper,
	svsl_emit_barrier_wg, svsl_emit_barrier_wg_mem,
	svsl_emit_barrier_device, svsl_emit_barrier_all,
	svsl_emit_barrier_device_sync, svsl_emit_barrier_all_sync,
	svsl_emit_subgroup,      // routes to the subgroup/quad sub-dispatcher (parses the suffix)
	svsl_emit_builtin_var,   // load a builtin input variable (WaveGetLaneCount → subgroup_size)
} svsl_emit_;

typedef struct svsl_intrinsic_t {
	const char *name;
	uint8_t     params[5]; // svsl_iparam_, 0-terminated
	uint8_t     result;    // svsl_ires_
	const char *opt_native; // porting hint ("use X"), NULL when this is the native form
	uint8_t     tag;        // svsl_intr_tag_
	uint8_t     emit;       // svsl_emit_
	uint16_t    op[3];      // ext450/core opcodes: float/signed/unsigned (op[1]==0 → float-only)
} svsl_intrinsic_t;

// returns table index or -1
int32_t                 svsl_intrinsic_find(svsl_str_t name);
const svsl_intrinsic_t *svsl_intrinsic_get (int32_t index);
const char             *svsl_intrinsic_get_name(int32_t index);

// read-only builtin variables (subgroup_size, ...): returns uint-typed builtin id or -1
int32_t svsl_builtin_var_find(svsl_str_t name);

// resource methods (tex.Sample, img.Store, ...)
typedef enum svsl_method_ {
	svsl_method_sample = 0,
	svsl_method_sample_level,
	svsl_method_sample_bias,
	svsl_method_sample_grad,
	svsl_method_sample_cmp,
	svsl_method_sample_cmp_level_zero,
	svsl_method_gather,       // + per-channel variants share this id, b = channel
	svsl_method_load,
	svsl_method_store,
	svsl_method_get_dimensions,
	svsl_method_atomic,       // image atomics: b = atomic op
	svsl_method_query_lod,    // CalculateLevelOfDetail: b = 1 for the unclamped variant
} svsl_method_;

int32_t svsl_method_find(svsl_str_t name, int32_t *out_channel_or_op); // svsl_method_ or -1
