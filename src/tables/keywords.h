// Keyword identifiers. Keywords are carried on identifier tokens (context-sensitive
// words like 'sample' or 'register' can still act as plain identifiers in the parser).

#pragma once

#include "../util/str.h"

#include <stdint.h>

typedef enum svsl_kw_ {
	svsl_kw_none = 0,
	// control flow
	svsl_kw_if, svsl_kw_else, svsl_kw_for, svsl_kw_while, svsl_kw_do,
	svsl_kw_switch, svsl_kw_case, svsl_kw_default,
	svsl_kw_break, svsl_kw_continue, svsl_kw_return, svsl_kw_discard, svsl_kw_demote,
	// declarations
	svsl_kw_struct, svsl_kw_enum, svsl_kw_cbuffer, svsl_kw_uniform, svsl_kw_storagebuffer,
	svsl_kw_pushconstant, svsl_kw_workgroup, svsl_kw_groupshared,
	svsl_kw_static, svsl_kw_const, svsl_kw_specialization, svsl_kw_register,
	svsl_kw_include,
	// parameter direction
	svsl_kw_in, svsl_kw_out, svsl_kw_inout,
	// access modifiers
	svsl_kw_readonly, svsl_kw_writeonly, svsl_kw_coherent, svsl_kw_volatile,
	// interpolation
	svsl_kw_flat, svsl_kw_noperspective, svsl_kw_centroid, svsl_kw_sample,
	svsl_kw_nointerpolation, svsl_kw_invariant, svsl_kw_precise,
	// layout
	svsl_kw_pack1, svsl_kw_pack8, svsl_kw_pack16,
	// inline SPIR-V
	svsl_kw_spirv,
	// literals
	svsl_kw_true, svsl_kw_false,
} svsl_kw_;

svsl_kw_ svsl_keyword_lookup(svsl_str_t ident);
