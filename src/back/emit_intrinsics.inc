// Included by emit_spirv.c: per-instruction emission and intrinsic dispatch.

// splatted constant of a value's shape (zeros / ones for bool conversions)
static uint32_t const_splat(emit_t *e, svsl_type_id_t shape, uint64_t scalar_bits) {
	const svsl_type_t *t  = svsl_type_get(&e->prog->types, shape);
	uint32_t           st = spv_scalar_type(e, t->scalar);
	bool wide  = svsl_scalar_size(t->scalar) == 8;
	bool bl    = t->scalar == svsl_scalar_bool;
	uint32_t c = svsl_spv_const(&e->spv, st, scalar_bits, wide, bl);
	if (t->kind != svsl_type_vector) return c;
	uint32_t id = svsl_spv_id(&e->spv);
	uint32_t words[6] = { spv_type_for(e, shape), id, c, c, c, c };
	svsl_spv_inst(&e->spv, &e->spv.types, SpvOpConstantComposite, words, 2 + (uint32_t)t->count);
	return id;
}

static uint32_t float_splat(emit_t *e, svsl_type_id_t shape, float v) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, shape);
	uint64_t bits;
	if (t->scalar == svsl_scalar_float16) {
		bits = svsl_f32_to_f16_bits(v);
	} else if (t->scalar == svsl_scalar_float64) {
		double d = (double)v;
		memcpy(&bits, &d, 8);
	} else {
		uint32_t b32;
		memcpy(&b32, &v, 4);
		bits = b32;
	}
	return const_splat(e, shape, bits);
}

// SpecConstantOp-legal opcodes under the Shader capability: integer and bool
// only, which also means float paths (FAdd, FOrd*, ...) can never route here
static bool spec_op_allowed(SpvOp op) {
	switch (op) {
	case SpvOpIAdd: case SpvOpISub: case SpvOpIMul:
	case SpvOpUDiv: case SpvOpSDiv: case SpvOpUMod: case SpvOpSRem:
	case SpvOpSNegate: case SpvOpNot:
	case SpvOpShiftLeftLogical: case SpvOpShiftRightLogical: case SpvOpShiftRightArithmetic:
	case SpvOpBitwiseAnd: case SpvOpBitwiseOr: case SpvOpBitwiseXor:
	case SpvOpLogicalAnd: case SpvOpLogicalOr: case SpvOpLogicalNot:
	case SpvOpLogicalEqual: case SpvOpLogicalNotEqual: case SpvOpSelect:
	case SpvOpIEqual: case SpvOpINotEqual:
	case SpvOpULessThan: case SpvOpSLessThan:
	case SpvOpUGreaterThan: case SpvOpSGreaterThan:
	case SpvOpULessThanEqual: case SpvOpSLessThanEqual:
	case SpvOpUGreaterThanEqual: case SpvOpSGreaterThanEqual:
		return true;
	default:
		return false;
	}
}

// emits `o` over the given IR operands — as OpSpecConstantOp in the global
// section when every operand is a constant and at least one depends on a
// specialization constant, so the expression stays specializable at pipeline
// creation instead of becoming runtime math
static uint32_t emit_op_or_spec(emit_t *e, svsl_spv_stream_t *fs, int32_t inst_index,
                                SpvOp o, uint32_t type_id,
                                const uint32_t *ir_operands, uint32_t count) {
	uint32_t ops[4];
	uint8_t  lo = 2, hi = 0;
	for (uint32_t k = 0; k < count; k++) {
		ops[k] = e->value_ids[ir_operands[k]];
		uint8_t g = e->value_spec[ir_operands[k]];
		if (g < lo) lo = g;
		if (g > hi) hi = g;
	}
	if (lo >= 1 && hi == 2 && spec_op_allowed(o)) {
		uint32_t id = svsl_spv_id(&e->spv);
		uint32_t words[8] = { type_id, id, (uint32_t)o };
		for (uint32_t k = 0; k < count; k++) words[3 + k] = ops[k];
		svsl_spv_inst(&e->spv, &e->spv.types, SpvOpSpecConstantOp, words, 3 + count);
		e->value_spec[inst_index] = 2;
		return id;
	}
	return emit_value_inst(e, fs, o, type_id, ops, count);
}

// --- intrinsic dispatch --------------------------------------------------------

// subgroup arithmetic family: opcode by scalar family
typedef struct sg_arith_t { const char *suffix; uint16_t f, s, u; } sg_arith_t;
static const sg_arith_t sg_arith[] = {
	{ "add", SpvOpGroupNonUniformFAdd, SpvOpGroupNonUniformIAdd, SpvOpGroupNonUniformIAdd },
	{ "mul", SpvOpGroupNonUniformFMul, SpvOpGroupNonUniformIMul, SpvOpGroupNonUniformIMul },
	{ "min", SpvOpGroupNonUniformFMin, SpvOpGroupNonUniformSMin, SpvOpGroupNonUniformUMin },
	{ "max", SpvOpGroupNonUniformFMax, SpvOpGroupNonUniformSMax, SpvOpGroupNonUniformUMax },
	{ "and", 0, SpvOpGroupNonUniformBitwiseAnd, SpvOpGroupNonUniformBitwiseAnd },
	{ "or",  0, SpvOpGroupNonUniformBitwiseOr,  SpvOpGroupNonUniformBitwiseOr },
	{ "xor", 0, SpvOpGroupNonUniformBitwiseXor, SpvOpGroupNonUniformBitwiseXor },
};

// rebuilds a composite value between its laid-out (buffer) and plain value
// types, member by member (to_value: laid → value; else value → laid)
static uint32_t convert_composite(emit_t *e, uint32_t value, svsl_type_id_t type,
                                  svsl_layout_ layout, bool to_value) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	uint32_t value_t = spv_type_for(e, type);
	uint32_t laid_t  = spv_type_laid(e, type, layout);
	if (value_t == laid_t) return value; // scalars/vectors/matrices are shared

	uint32_t  stack_parts[64];
	uint32_t *parts = stack_parts;
	int32_t   count = 0;
	if (t->kind == svsl_type_struct) {
		const svsl_struct_info_t *info = &e->prog->types.structs.items[t->struct_index];
		count = info->members.count;
		if (count > 64) parts = svsl_arena_alloc(e->arena, (size_t)count * 4);
		for (int32_t m = 0; m < count; m++) {
			svsl_type_id_t mt = info->members.items[m].type;
			uint32_t part_t = to_value ? spv_type_laid(e, mt, layout) : spv_type_for(e, mt);
			uint32_t part = emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract, part_t,
			                                (uint32_t[]){ value, (uint32_t)m }, 2);
			parts[m] = convert_composite(e, part, mt, layout, to_value);
		}
	} else if (t->kind == svsl_type_array && t->array_count > 0) {
		count = t->array_count;
		if (count > 64) parts = svsl_arena_alloc(e->arena, (size_t)count * 4);
		for (int32_t m = 0; m < count; m++) {
			uint32_t part_t = to_value ? spv_type_laid(e, t->elem, layout) : spv_type_for(e, t->elem);
			uint32_t part = emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract, part_t,
			                                (uint32_t[]){ value, (uint32_t)m }, 2);
			parts[m] = convert_composite(e, part, t->elem, layout, to_value);
		}
	} else if (t->scalar == svsl_scalar_bool) {
		// bools live in buffers as uint 0/1 (spv_type_laid): != 0 on the way to a
		// value, select 1/0 on the way back
		svsl_type_id_t ushape = bool_buffer_uint(e, t);
		if (to_value)
			return emit_value_inst(e, &e->spv.funcs, SpvOpINotEqual, value_t,
			                       (uint32_t[]){ value, const_splat(e, ushape, 0) }, 2);
		return emit_value_inst(e, &e->spv.funcs, SpvOpSelect, laid_t,
		                       (uint32_t[]){ value, const_splat(e, ushape, 1),
		                                     const_splat(e, ushape, 0) }, 3);
	} else {
		return value;
	}
	return emit_value_inst(e, &e->spv.funcs, SpvOpCompositeConstruct,
	                       to_value ? value_t : laid_t, parts, (uint32_t)count);
}

static uint32_t emit_ext450(emit_t *e, uint32_t type_id, uint32_t number,
                            const uint32_t *args, uint32_t count) {
	uint32_t ops[12] = { e->spv.glsl450, number };
	for (uint32_t i = 0; i < count && i < 10; i++) ops[2 + i] = args[i];
	return emit_value_inst(e, &e->spv.funcs, SpvOpExtInst, type_id, ops, count + 2);
}

static uint32_t emit_intrinsic(emit_t *e, int32_t index, const svsl_ir_inst_t *inst) {
	const svsl_intrinsic_t *intr = svsl_intrinsic_get((int32_t)inst->args[0]);
	const char *name = intr->opt_native ? intr->opt_native : intr->name;
	(void)index;

	uint32_t args[8];
	uint32_t argc = inst->aux_count < 8 ? inst->aux_count : 8;
	for (uint32_t i = 0; i < argc; i++)
		args[i] = e->value_ids[e->fn->aux.items[inst->aux + i]];
	uint32_t type_id = spv_type_for(e, inst->type);

	svsl_scalar_ family = svsl_scalar_float32;
	if (inst->aux_count > 0)
		family = value_scalar(e, e->fn->aux.items[inst->aux]);
	bool floaty = scalar_is_floaty(family);
	bool sgn    = scalar_is_signed(family);

	switch ((svsl_emit_)intr->emit) {
	case svsl_emit_ext450: { // GLSLstd450 ext inst, float/signed/unsigned variant
		uint16_t number = intr->op[1] == 0 ? intr->op[0]
		                : floaty ? intr->op[0] : (sgn ? intr->op[1] : intr->op[2]);
		if (number == 0) return args[0]; // e.g. abs(uint) is the identity
		return emit_ext450(e, type_id, number, args, argc);
	}
	case svsl_emit_core: { // plain SpvOp (derivatives detected by opcode range)
		SpvOp op = (SpvOp)intr->op[0];
		if (op == SpvOpDPdxCoarse || op == SpvOpDPdyCoarse ||
		    op == SpvOpDPdxFine || op == SpvOpDPdyFine)
			svsl_spv_cap(&e->spv, SpvCapabilityDerivativeControl);
		// derivatives require 32-bit components: float16 operands round-trip
		// through float32 (like DXC)
		if (family == svsl_scalar_float16 && op >= SpvOpDPdx && op <= SpvOpFwidthCoarse) {
			svsl_types_t      *types = (svsl_types_t *)&e->prog->types;
			const svsl_type_t *rt2   = svsl_type_get(types, inst->type);
			svsl_type_id_t f32_shape = rt2->kind == svsl_type_vector
				? svsl_type_vector_id(types, svsl_scalar_float32, rt2->count)
				: svsl_type_scalar_id(types, svsl_scalar_float32);
			uint32_t f32_id = spv_type_for(e, f32_shape);
			uint32_t wide   = emit_value_inst(e, &e->spv.funcs, SpvOpFConvert, f32_id,
			                                  (uint32_t[]){ args[0] }, 1);
			uint32_t deriv  = emit_value_inst(e, &e->spv.funcs, op, f32_id,
			                                  (uint32_t[]){ wide }, 1);
			return emit_value_inst(e, &e->spv.funcs, SpvOpFConvert, type_id,
			                       (uint32_t[]){ deriv }, 1);
		}
		return emit_value_inst(e, &e->spv.funcs, op, type_id, args, argc);
	}

	case svsl_emit_bitfield_extract: { // (base, offset, count)
		SpvOp op = sgn ? SpvOpBitFieldSExtract : SpvOpBitFieldUExtract;
		return emit_value_inst(e, &e->spv.funcs, op, type_id, args, argc);
	}
	case svsl_emit_bitfield_insert: // (base, insert, offset, count)
		return emit_value_inst(e, &e->spv.funcs, SpvOpBitFieldInsert, type_id, args, argc);
	case svsl_emit_rcp: {
		uint32_t one = float_splat(e, inst->type, 1.0f);
		return emit_value_inst(e, &e->spv.funcs, SpvOpFDiv, type_id, (uint32_t[]){ one, args[0] }, 2);
	}
	case svsl_emit_ldexp: { // HLSL takes a float exponent: x * exp2(e)
		uint32_t p2 = emit_ext450(e, type_id, GLSLstd450Exp2, &args[1], 1);
		return emit_value_inst(e, &e->spv.funcs, SpvOpFMul, type_id, (uint32_t[]){ args[0], p2 }, 2);
	}
	case svsl_emit_frexp_mant:
	case svsl_emit_frexp_exp: {
		// FrexpStruct returns { mantissa : G, exponent : int-shaped-G }
		svsl_types_t      *types = (svsl_types_t *)&e->prog->types;
		const svsl_type_t *rt    = svsl_type_get(types, inst->type);
		int32_t  n      = rt->kind == svsl_type_vector ? rt->count : 1;
		svsl_type_id_t g = e->fn->insts.items[e->fn->aux.items[inst->aux]].type;
		uint32_t int_t  = spv_type_for(e, n == 1 ? svsl_type_scalar_id(types, svsl_scalar_int32)
		                                         : svsl_type_vector_id(types, svsl_scalar_int32, n));
		uint32_t pair_t = svsl_spv_type(&e->spv, SpvOpTypeStruct,
		                                (uint32_t[]){ spv_type_for(e, g), int_t }, 2);
		uint32_t pair   = emit_ext450(e, pair_t, GLSLstd450FrexpStruct, args, 1);
		if (intr->emit == svsl_emit_frexp_mant)
			return emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract, spv_type_for(e, g),
			                       (uint32_t[]){ pair, 0 }, 2);
		uint32_t exp_i = emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract, int_t,
		                                 (uint32_t[]){ pair, 1 }, 2);
		return emit_value_inst(e, &e->spv.funcs, SpvOpConvertSToF, type_id, &exp_i, 1);
	}
	case svsl_emit_saturate: {
		uint32_t ops[3] = { args[0],
		                    float_splat(e, inst->type, 0.0f),
		                    float_splat(e, inst->type, 1.0f) };
		return emit_ext450(e, type_id, GLSLstd450FClamp, ops, 3);
	}
	case svsl_emit_log10: {
		uint32_t lg = emit_ext450(e, type_id, GLSLstd450Log2, args, 1);
		uint32_t k  = float_splat(e, inst->type, 0.30102999566f);
		return emit_value_inst(e, &e->spv.funcs, SpvOpFMul, type_id, (uint32_t[]){ lg, k }, 2);
	}
	case svsl_emit_any:
	case svsl_emit_all: {
		svsl_type_id_t at = e->fn->insts.items[e->fn->aux.items[inst->aux]].type;
		if (svsl_type_get(&e->prog->types, at)->kind != svsl_type_vector)
			return args[0]; // scalar bool passes through
		return emit_value_inst(e, &e->spv.funcs,
		                       intr->emit == svsl_emit_any ? SpvOpAny : SpvOpAll, type_id, args, 1);
	}
	case svsl_emit_bitcast: { // asfloat/asuint/asint
		svsl_type_id_t at = e->fn->insts.items[e->fn->aux.items[inst->aux]].type;
		if (spv_type_for(e, at) == type_id) return args[0];
		return emit_value_inst(e, &e->spv.funcs, SpvOpBitcast, type_id, args, 1);
	}
	case svsl_emit_f16tof32:
	case svsl_emit_f32tof16: {
		bool to32 = intr->emit == svsl_emit_f16tof32;
		svsl_type_id_t at = e->fn->insts.items[e->fn->aux.items[inst->aux]].type;
		const svsl_type_t *t = svsl_type_get(&e->prog->types, at);
		int32_t n = t->kind == svsl_type_vector ? t->count : 1;
		uint32_t f32   = spv_scalar_type(e, svsl_scalar_float32);
		uint32_t u32   = spv_scalar_type(e, svsl_scalar_uint32);
		uint32_t vec2f = svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ f32, 2 }, 2);
		uint32_t parts[4];
		for (int32_t i = 0; i < n; i++) {
			uint32_t comp = n == 1 ? args[0]
			              : emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract,
			                                to32 ? u32 : f32, (uint32_t[]){ args[0], (uint32_t)i }, 2);
			if (to32) { // unpackHalf2x16(x).x
				uint32_t pair = emit_ext450(e, vec2f, GLSLstd450UnpackHalf2x16, &comp, 1);
				parts[i] = emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract, f32,
				                           (uint32_t[]){ pair, 0 }, 2);
			} else {    // packHalf2x16(vec2(x, 0))
				uint32_t zero = spv_float_const(e, 0.0f);
				uint32_t pair = emit_value_inst(e, &e->spv.funcs, SpvOpCompositeConstruct, vec2f,
				                                (uint32_t[]){ comp, zero }, 2);
				parts[i] = emit_ext450(e, u32, GLSLstd450PackHalf2x16, &pair, 1);
			}
		}
		if (n == 1) return parts[0];
		return emit_value_inst(e, &e->spv.funcs, SpvOpCompositeConstruct, type_id,
		                       parts, (uint32_t)n);
	}
	case svsl_emit_clip: {
		svsl_type_id_t     at = e->fn->insts.items[e->fn->aux.items[inst->aux]].type;
		const svsl_type_t *t  = svsl_type_get(&e->prog->types, at);
		int32_t n = t->kind == svsl_type_vector ? t->count : 1;
		uint32_t boolt = svsl_spv_type(&e->spv, SpvOpTypeBool, NULL, 0);
		uint32_t cmp_type = n == 1 ? boolt
		                  : svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ boolt, (uint32_t)n }, 2);
		uint32_t zero = float_splat(e, at, 0.0f);
		uint32_t cmp  = emit_value_inst(e, &e->spv.funcs, SpvOpFOrdLessThan, cmp_type,
		                                (uint32_t[]){ args[0], zero }, 2);
		if (n > 1) cmp = emit_value_inst(e, &e->spv.funcs, SpvOpAny, boolt, &cmp, 1);
		uint32_t kill  = svsl_spv_id(&e->spv);
		uint32_t merge = svsl_spv_id(&e->spv);
		svsl_spv_inst2(&e->spv, &e->spv.funcs, SpvOpSelectionMerge, merge, SpvSelectionControlMaskNone);
		svsl_spv_inst3(&e->spv, &e->spv.funcs, SpvOpBranchConditional, cmp, kill, merge);
		begin_block(e, kill);
		svsl_spv_inst(&e->spv, &e->spv.funcs, SpvOpKill, NULL, 0);
		begin_block(e, merge);
		return 0;
	}
	case svsl_emit_tile_depth:
	case svsl_emit_tile_stencil: {
		// SPV_EXT_shader_tile_image: read this fragment's depth/stencil in tile memory
		bool depth = intr->emit == svsl_emit_tile_depth;
		svsl_spv_cap(&e->spv, depth ? SpvCapabilityTileImageDepthReadAccessEXT
		                            : SpvCapabilityTileImageStencilReadAccessEXT);
		svsl_spv_extension(&e->spv, "SPV_EXT_shader_tile_image");
		return emit_value_inst(e, &e->spv.funcs,
		                       depth ? SpvOpDepthAttachmentReadEXT : SpvOpStencilAttachmentReadEXT,
		                       type_id, NULL, 0);
	}
	case svsl_emit_is_helper: {
		// OpIsHelperInvocationEXT, not the HelperInvocation builtin: the builtin
		// variable's value is undefined once the invocation demotes
		uint32_t boolt = svsl_spv_type(&e->spv, SpvOpTypeBool, NULL, 0);
		svsl_spv_cap(&e->spv, SpvCapabilityDemoteToHelperInvocationEXT);
		svsl_spv_extension(&e->spv, "SPV_EXT_demote_to_helper_invocation");
		return emit_value_inst(e, &e->spv.funcs, SpvOpIsHelperInvocationEXT, boolt, NULL, 0);
	}

	case svsl_emit_barrier_wg: {
		uint32_t scope = spv_uint_const(e, SpvScopeWorkgroup);
		uint32_t sem   = spv_uint_const(e, SpvMemorySemanticsAcquireReleaseMask |
		                                   SpvMemorySemanticsWorkgroupMemoryMask);
		svsl_spv_inst3(&e->spv, &e->spv.funcs, SpvOpControlBarrier, scope, scope, sem);
		return 0;
	}
	case svsl_emit_barrier_wg_mem: {
		uint32_t scope = spv_uint_const(e, SpvScopeWorkgroup);
		uint32_t sem   = spv_uint_const(e, SpvMemorySemanticsAcquireReleaseMask |
		                                   SpvMemorySemanticsWorkgroupMemoryMask);
		svsl_spv_inst2(&e->spv, &e->spv.funcs, SpvOpMemoryBarrier, scope, sem);
		return 0;
	}
	// device/all memory semantics match glslang: device adds Uniform+Image,
	// "all" additionally Workgroup+AtomicCounter; _sync variants are
	// OpControlBarrier with Workgroup execution scope (verified vs skshaderc)
	case svsl_emit_barrier_device:
	case svsl_emit_barrier_all:
	case svsl_emit_barrier_device_sync:
	case svsl_emit_barrier_all_sync: {
		bool     all   = intr->emit == svsl_emit_barrier_all || intr->emit == svsl_emit_barrier_all_sync;
		bool     sync  = intr->emit == svsl_emit_barrier_device_sync || intr->emit == svsl_emit_barrier_all_sync;
		uint32_t scope = spv_uint_const(e, SpvScopeDevice);
		uint32_t sem   = spv_uint_const(e, SpvMemorySemanticsAcquireReleaseMask |
		                                   SpvMemorySemanticsUniformMemoryMask |
		                                   SpvMemorySemanticsImageMemoryMask |
		                                   (all ? SpvMemorySemanticsWorkgroupMemoryMask |
		                                          SpvMemorySemanticsAtomicCounterMemoryMask : 0));
		if (sync)
			svsl_spv_inst3(&e->spv, &e->spv.funcs, SpvOpControlBarrier,
			               spv_uint_const(e, SpvScopeWorkgroup), scope, sem);
		else
			svsl_spv_inst2(&e->spv, &e->spv.funcs, SpvOpMemoryBarrier, scope, sem);
		return 0;
	}

	case svsl_emit_subgroup: {
		svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniform);
		uint32_t scope = spv_uint_const(e, SpvScopeSubgroup);
		const char *sub = name + (name[0] == 'q' ? 5 : 9);

		if (strcmp(name, "subgroup_elect") == 0)
			return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformElect, type_id, &scope, 1);
		if (strcmp(sub, "all") == 0 || strcmp(sub, "any") == 0 || strcmp(sub, "all_equal") == 0) {
			svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformVote);
			SpvOp op = sub[1] == 'n' ? SpvOpGroupNonUniformAny :
			           sub[3] == '\0' ? SpvOpGroupNonUniformAll : SpvOpGroupNonUniformAllEqual;
			return emit_value_inst(e, &e->spv.funcs, op, type_id, (uint32_t[]){ scope, args[0] }, 2);
		}
		if (strcmp(sub, "ballot") == 0) {
			svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformBallot);
			return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformBallot, type_id,
			                       (uint32_t[]){ scope, args[0] }, 2);
		}
		if (strcmp(sub, "ballot_bit_count") == 0 || strcmp(sub, "ballot_exclusive_bit_count") == 0) {
			svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformBallot);
			uint32_t u32   = spv_scalar_type(e, svsl_scalar_uint32);
			uint32_t uvec4 = svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ u32, 4 }, 2);
			uint32_t ballot = emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformBallot, uvec4,
			                                  (uint32_t[]){ scope, args[0] }, 2);
			uint32_t group_op = sub[7] == 'e' ? SpvGroupOperationExclusiveScan
			                                  : SpvGroupOperationReduce;
			return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformBallotBitCount, type_id,
			                       (uint32_t[]){ scope, group_op, ballot }, 3);
		}
		if (strcmp(sub, "broadcast") == 0 || strcmp(sub, "broadcast_first") == 0) {
			svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformBallot);
			if (sub[9] == '\0')
				return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformBroadcast, type_id,
				                       (uint32_t[]){ scope, args[0], args[1] }, 3);
			return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformBroadcastFirst, type_id,
			                       (uint32_t[]){ scope, args[0] }, 2);
		}
		if (strncmp(sub, "shuffle", 7) == 0) {
			SpvOp op = SpvOpGroupNonUniformShuffle;
			if      (strcmp(sub, "shuffle_xor") == 0)  op = SpvOpGroupNonUniformShuffleXor;
			else if (strcmp(sub, "shuffle_up") == 0)   op = SpvOpGroupNonUniformShuffleUp;
			else if (strcmp(sub, "shuffle_down") == 0) op = SpvOpGroupNonUniformShuffleDown;
			svsl_spv_cap(&e->spv, op == SpvOpGroupNonUniformShuffle || op == SpvOpGroupNonUniformShuffleXor
			                      ? SpvCapabilityGroupNonUniformShuffle
			                      : SpvCapabilityGroupNonUniformShuffleRelative);
			return emit_value_inst(e, &e->spv.funcs, op, type_id,
			                       (uint32_t[]){ scope, args[0], args[1] }, 3);
		}
		if (name[0] == 'q') { // quad_*
			svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformQuad);
			if (strcmp(sub, "broadcast") == 0)
				return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformQuadBroadcast, type_id,
				                       (uint32_t[]){ scope, args[0], args[1] }, 3);
			uint32_t dir = strcmp(sub, "swap_horizontal") == 0 ? 0u :
			               strcmp(sub, "swap_vertical") == 0 ? 1u : 2u;
			return emit_value_inst(e, &e->spv.funcs, SpvOpGroupNonUniformQuadSwap, type_id,
			                       (uint32_t[]){ scope, args[0], spv_uint_const(e, dir) }, 3);
		}
		// arithmetic: [inclusive_|exclusive_|clustered_]<op>
		uint32_t group_op = SpvGroupOperationReduce;
		if      (strncmp(sub, "inclusive_", 10) == 0) { group_op = SpvGroupOperationInclusiveScan; sub += 10; }
		else if (strncmp(sub, "exclusive_", 10) == 0) { group_op = SpvGroupOperationExclusiveScan; sub += 10; }
		else if (strncmp(sub, "clustered_", 10) == 0) { group_op = SpvGroupOperationClusteredReduce; sub += 10; }
		for (int32_t i = 0; i < (int32_t)(sizeof(sg_arith) / sizeof(sg_arith[0])); i++) {
			if (strcmp(sub, sg_arith[i].suffix) != 0) continue;
			svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformArithmetic);
			if (group_op == SpvGroupOperationClusteredReduce)
				svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniformClustered);
			uint16_t op = floaty ? sg_arith[i].f : (sgn ? sg_arith[i].s : sg_arith[i].u);
			uint32_t ops[4] = { scope, group_op, args[0], args[1] };
			return emit_value_inst(e, &e->spv.funcs, (SpvOp)op, type_id, ops,
			                       group_op == SpvGroupOperationClusteredReduce ? 4u : 3u);
		}
		break; // unknown suffix → error below (shouldn't happen for a table row)
	}
	case svsl_emit_builtin_var: {
		// zero-arg intrinsic whose native name is a builtin input variable
		// (WaveGetLaneCount → subgroup_size, tile_offset_qcom, …): load it
		int32_t builtin = svsl_builtin_var_find(svsl_str(name));
		if (builtin >= 0) {
			if (builtin >= 4 && e->fn->entry->stage == svsl_stage_vertex) { // tile builtins
				eerr(e, inst->loc, "tile shading builtins need a pixel or compute shader");
				return 0;
			}
			uint32_t u32   = spv_scalar_type(e, svsl_scalar_uint32);
			uint32_t comps = builtin_input_comps((uint32_t)builtin);
			uint32_t tid   = comps == 1 ? u32
			               : svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ u32, comps }, 2);
			return emit_value_inst(e, &e->spv.funcs, SpvOpLoad, tid,
			                       (uint32_t[]){ builtin_input_var(e, (uint32_t)builtin) }, 1);
		}
		break;
	}
	case svsl_emit_none:
		break; // lowered before emit, or unmapped (e.g. select) → error
	}

	eerr(e, inst->loc, "intrinsic has no SPIR-V mapping yet");
	return 0;
}

// --- the per-instruction switch ------------------------------------------------------

// Inline SPIR-V (spirv_asm): emit the block's instructions verbatim, splicing in
// SVSL values ($ → value_ids, consumed from aux in binary order), SVSL types
// ($$ → spv_type_for), and named locals (%id → fresh SPIR-V ids). The %result
// local carries the block's value.
static uint32_t emit_spirv_asm(emit_t *e, const svsl_ir_inst_t *inst) {
	const svsl_ast_expr_t *node = e->fn->asm_nodes.items[inst->args[0]];
	struct { svsl_str_t name; uint32_t id; } locals[64];
	int32_t  local_count = 0;
	uint32_t vcur        = 0; // cursor into aux ($value ids, in binary order)

	for (int32_t k = 0; k < node->spirv_asm.inst_count; k++) {
		const svsl_ast_spv_inst_t *ai = &node->spirv_asm.insts[k];
		if (ai->spv_op == SpvOpCapability) { // requirements route to the module's
			svsl_spv_cap(&e->spv, (SpvCapability)ai->operands[0].literal);
			continue;                        // declaration streams, not the code
		}
		if (ai->spv_op == SpvOpExtension) {
			svsl_spv_extension(&e->spv, svsl_arena_strndup(e->arena, ai->operands[0].string.ptr,
			                                               (size_t)ai->operands[0].string.len));
			continue;
		}
		uint32_t words[128];
		uint32_t count = 0;
		for (int32_t o = 0; o < ai->operand_count && count < 128; o++) {
			const svsl_ast_spv_operand_t *op = &ai->operands[o];
			if (op->kind == svsl_spv_operand_string) { // packs to multiple words
				uint32_t sw = ((uint32_t)op->string.len + 4) / 4; // incl. nul + pad
				if (count + sw > 128) { eerr(e, op->loc, "spirv_asm instruction too long"); return 0; }
				words[count + sw - 1] = 0;
				memcpy(&words[count], op->string.ptr, (size_t)op->string.len);
				count += sw;
				continue;
			}
			uint32_t w = 0;
			switch ((svsl_spv_operand_)op->kind) {
			case svsl_spv_operand_local: {
				int32_t f = -1;
				for (int32_t l = 0; l < local_count; l++)
					if (svsl_str_eq(locals[l].name, op->local)) { f = l; break; }
				if (f < 0) {
					if (local_count >= 64) { eerr(e, op->loc, "too many spirv_asm result ids"); return 0; }
					f = local_count++;
					locals[f].name = op->local;
					locals[f].id   = svsl_spv_id(&e->spv);
				}
				w = locals[f].id;
				break;
			}
			case svsl_spv_operand_value:   w = e->value_ids[e->fn->aux.items[inst->aux + vcur++]]; break;
			case svsl_spv_operand_type:    w = spv_type_for(e, op->type_id);                       break;
			case svsl_spv_operand_literal: w = op->literal;                                         break;
			case svsl_spv_operand_glsl450: w = e->spv.glsl450;                                      break;
			case svsl_spv_operand_string:  break; // packed above
			}
			words[count++] = w;
		}
		svsl_spv_inst(&e->spv, &e->spv.funcs, (SpvOp)ai->spv_op, words, count);
	}

	for (int32_t l = 0; l < local_count; l++)
		if (svsl_str_eq_cstr(locals[l].name, "result")) return locals[l].id;
	return 0;
}

static void emit_inst(emit_t *e, int32_t i, const svsl_ir_inst_t *inst,
                      const svsl_type_t *rt, svsl_type_id_t ret_type) {
	svsl_spv_t        *spv = &e->spv;
	svsl_spv_stream_t *fs  = &spv->funcs;
	svsl_ir_op_        op  = (svsl_ir_op_)inst->op;
	uint32_t           result = 0;

	// control flow first (block bookkeeping)
	switch (op) {
	case svsl_ir_if: {
		ensure_block(e);
		uint32_t merge = svsl_spv_id(spv);
		uint32_t then  = svsl_spv_id(spv);
		uint32_t other = e->if_has_else[i] ? svsl_spv_id(spv) : merge;
		svsl_spv_inst2(spv, fs, SpvOpSelectionMerge, merge, inst->args[1]); // [flatten]/[branch] hint
		svsl_spv_inst3(spv, fs, SpvOpBranchConditional, e->value_ids[inst->args[0]], then, other);
		e->terminated = true;
		e->cf[e->cf_depth++] = (struct cf_frame){ .kind = 'i', .merge = merge,
		                                          .cont = e->if_has_else[i] ? other : 0 };
		begin_block(e, then);
		return;
	}
	case svsl_ir_else: {
		struct cf_frame *f = &e->cf[e->cf_depth - 1];
		branch_to(e, f->merge);
		begin_block(e, f->cont);
		return;
	}
	case svsl_ir_end_if: {
		struct cf_frame *f = &e->cf[--e->cf_depth];
		branch_to(e, f->merge);
		begin_block(e, f->merge);
		return;
	}
	case svsl_ir_loop: {
		ensure_block(e);
		uint32_t header = svsl_spv_id(spv);
		uint32_t body   = svsl_spv_id(spv);
		uint32_t cont   = svsl_spv_id(spv);
		uint32_t merge  = svsl_spv_id(spv);
		branch_to(e, header);
		begin_block(e, header);
		svsl_spv_inst3(spv, fs, SpvOpLoopMerge, merge, cont, inst->args[0]); // [unroll]/[loop] hint
		svsl_spv_inst1(spv, fs, SpvOpBranch, body);
		e->terminated = true;
		uint32_t *loop_hdr = svsl_arena_alloc(e->arena, sizeof(uint32_t));
		loop_hdr[0] = header; // back-edge target for end_loop
		e->cf[e->cf_depth++] = (struct cf_frame){ .kind = 'l', .merge = merge, .cont = cont,
		                                          .case_labels = loop_hdr };
		begin_block(e, body);
		return;
	}
	case svsl_ir_loop_continue: {
		struct cf_frame *f = &e->cf[e->cf_depth - 1];
		f->cont_seen = true;
		branch_to(e, f->cont);
		begin_block(e, f->cont);
		return;
	}
	case svsl_ir_end_loop: {
		struct cf_frame *f = &e->cf[--e->cf_depth];
		if (!f->cont_seen) { // wrapper loops have no continue marker: synthesize the block
			branch_to(e, f->cont);
			begin_block(e, f->cont);
		}
		if (inst->args[0] != SVSL_IR_NONE && !e->terminated) { // do-while repeat condition
			svsl_spv_inst3(spv, fs, SpvOpBranchConditional,
			               e->value_ids[inst->args[0]], f->case_labels[0], f->merge);
			e->terminated = true;
		} else {
			branch_to(e, f->case_labels[0]); // back edge to the header
		}
		begin_block(e, f->merge);
		return;
	}
	case svsl_ir_break: {
		struct cf_frame *f = cf_innermost_breakable(e);
		if (f) branch_to(e, f->merge);
		return;
	}
	case svsl_ir_continue: {
		struct cf_frame *f = cf_innermost_loop(e);
		if (f) branch_to(e, f->cont);
		return;
	}
	case svsl_ir_switch: {
		ensure_block(e);
		struct cf_frame frame = { .kind = 's', .merge = svsl_spv_id(spv) };
		// count case markers belonging to this switch (they follow in order)
		int32_t depth = 0, ncases = 0;
		for (int32_t k = i + 1; k < e->fn->insts.count; k++) {
			svsl_ir_op_ ko = (svsl_ir_op_)e->fn->insts.items[k].op;
			if (ko == svsl_ir_switch) depth++;
			if (ko == svsl_ir_end_switch) { if (depth == 0) break; depth--; }
			if (ko == svsl_ir_case && depth == 0) ncases++;
		}
		frame.case_labels = svsl_arena_alloc(e->arena, (size_t)(ncases > 0 ? ncases : 1) * sizeof(uint32_t));
		depth = 0;
		for (int32_t k = i + 1; k < e->fn->insts.count; k++) {
			svsl_ir_op_ ko = (svsl_ir_op_)e->fn->insts.items[k].op;
			if (ko == svsl_ir_switch) depth++;
			if (ko == svsl_ir_end_switch) { if (depth == 0) break; depth--; }
			if (ko == svsl_ir_case && depth == 0)
				frame.case_labels[frame.case_count++] = svsl_spv_id(spv);
		}
		uint32_t default_label = frame.merge;
		{ // find the default case ordinal
			int32_t d2 = 0, ordinal = 0;
			for (int32_t k = i + 1; k < e->fn->insts.count; k++) {
				svsl_ir_op_ ko = (svsl_ir_op_)e->fn->insts.items[k].op;
				if (ko == svsl_ir_switch) d2++;
				if (ko == svsl_ir_end_switch) { if (d2 == 0) break; d2--; }
				if (ko == svsl_ir_case && d2 == 0) {
					if (e->fn->insts.items[k].args[1]) default_label = frame.case_labels[ordinal];
					ordinal++;
				}
			}
		}
		svsl_spv_inst2(spv, fs, SpvOpSelectionMerge, frame.merge, SpvSelectionControlMaskNone);
		uint32_t *words = svsl_arena_alloc(e->arena, (size_t)(2 + 2 * ncases) * sizeof(uint32_t));
		uint32_t count = 0;
		words[count++] = e->value_ids[inst->args[0]];
		words[count++] = default_label;
		{
			int32_t d2 = 0, ordinal = 0;
			for (int32_t k = i + 1; k < e->fn->insts.count; k++) {
				svsl_ir_op_ ko = (svsl_ir_op_)e->fn->insts.items[k].op;
				if (ko == svsl_ir_switch) d2++;
				if (ko == svsl_ir_end_switch) { if (d2 == 0) break; d2--; }
				if (ko == svsl_ir_case && d2 == 0) {
					if (!e->fn->insts.items[k].args[1]) {
						words[count++] = e->fn->aux.items[inst->aux + ordinal];
						words[count++] = frame.case_labels[ordinal];
					}
					ordinal++;
				}
			}
		}
		svsl_spv_inst(spv, fs, SpvOpSwitch, words, count);
		e->terminated = true;
		e->cf[e->cf_depth++] = frame;
		return;
	}
	case svsl_ir_case: {
		struct cf_frame *f = &e->cf[e->cf_depth - 1];
		uint32_t label = f->case_labels[f->next_case < f->case_count ? f->next_case : 0];
		f->next_case++;
		branch_to(e, label); // fallthrough from the previous case
		begin_block(e, label);
		return;
	}
	case svsl_ir_end_switch: {
		struct cf_frame *f = &e->cf[--e->cf_depth];
		branch_to(e, f->merge);
		begin_block(e, f->merge);
		return;
	}
	case svsl_ir_return: {
		ensure_block(e);
		// scalar-replaced output: members already stored straight to the interface
		bool sroa_out = e->output_sroa_var != SVSL_IR_NONE && inst->args[0] != SVSL_IR_NONE &&
		                e->fn->insts.items[inst->args[0]].op == svsl_ir_load &&
		                e->fn->insts.items[inst->args[0]].args[0] == e->output_sroa_var;
		if (!sroa_out && inst->args[0] != SVSL_IR_NONE && e->output_count > 0) {
			uint32_t value = e->value_ids[inst->args[0]];
			if (rt->kind == svsl_type_struct) {
				const svsl_struct_info_t *si = &e->prog->types.structs.items[rt->struct_index];
				for (int32_t m = 0; m < si->members.count && m < e->output_count; m++) {
					uint32_t part = emit_value_inst(e, fs, SpvOpCompositeExtract,
					                                spv_type_for(e, si->members.items[m].type),
					                                (uint32_t[]){ value, (uint32_t)m }, 2);
					svsl_spv_inst2(spv, fs, SpvOpStore, e->output_vars[m], part);
				}
			} else {
				svsl_spv_inst2(spv, fs, SpvOpStore, e->output_vars[0], value);
			}
			(void)ret_type;
		}
		svsl_spv_inst(spv, fs, SpvOpReturn, NULL, 0);
		e->terminated = true;
		return;
	}
	case svsl_ir_discard:
		ensure_block(e);
		svsl_spv_inst(spv, fs, SpvOpKill, NULL, 0);
		e->terminated = true;
		return;
	case svsl_ir_demote:
		ensure_block(e);
		svsl_spv_cap(spv, SpvCapabilityDemoteToHelperInvocationEXT);
		svsl_spv_extension(spv, "SPV_EXT_demote_to_helper_invocation");
		svsl_spv_inst(spv, fs, SpvOpDemoteToHelperInvocationEXT, NULL, 0);
		return;
	default:
		break;
	}

	ensure_block(e);
	uint32_t type_id = spv_type_for(e, inst->type);

	switch (op) {
	case svsl_ir_const: {
		const svsl_type_t *t = svsl_type_get(&e->prog->types, inst->type);
		uint64_t bits = (uint64_t)inst->args[0] | ((uint64_t)inst->args[1] << 32);
		result = svsl_spv_const(spv, type_id, bits, svsl_scalar_size(t->scalar) == 8,
		                        t->scalar == svsl_scalar_bool);
		e->value_spec[i] = 1;
		break;
	}
	case svsl_ir_spec_const: {
		uint32_t index = inst->args[0];
		if (!e->spec_const_ids[index]) {
			const svsl_spec_const_t *sc = &e->prog->spec_consts.items[index];
			uint32_t id = svsl_spv_id(spv);
			const svsl_type_t *t = svsl_type_get(&e->prog->types, sc->type);
			if (t->scalar == svsl_scalar_bool)
				svsl_spv_inst2(spv, &spv->types,
				               sc->default_bits ? SpvOpSpecConstantTrue : SpvOpSpecConstantFalse,
				               type_id, id);
			else
				svsl_spv_inst3(spv, &spv->types, SpvOpSpecConstant, type_id, id, sc->default_bits);
			svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, id, SpvDecorationSpecId, sc->id);
			svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ id }, 1, sc->name);
			e->spec_const_ids[index] = id;
		}
		result = e->spec_const_ids[index];
		e->value_spec[i] = 2;
		break;
	}
	case svsl_ir_undef:
		result = emit_value_inst(e, fs, SpvOpUndef, type_id, NULL, 0);
		break;
	case svsl_ir_ptr: {
		switch ((svsl_ref_)inst->args[0]) {
		case svsl_ref_buffer_member: {
			const svsl_buffer_t *buf = &e->prog->buffers.items[inst->args[1]];
			SpvStorageClass class_ =
				buf->kind == svsl_block_pushconstant ? SpvStorageClassPushConstant :
				buf->kind == svsl_block_storagebuffer ? SpvStorageClassStorageBuffer :
				SpvStorageClassUniform;
			uint32_t ptr   = spv_ptr_type(e, class_, spv_type_laid(e, inst->type, buf->layout));
			uint32_t index = spv_int_const(e, (int32_t)inst->args[2]);
			result = emit_value_inst(e, fs, SpvOpAccessChain, ptr,
			                         (uint32_t[]){ e->buffer_vars[inst->args[1]], index }, 2);
			e->value_class[i]  = class_;
			e->value_layout[i] = (uint8_t)buf->layout;
			break;
		}
		case svsl_ref_resource:
			result = e->resource_vars[inst->args[1]];
			e->value_class[i]  = SpvStorageClassStorageBuffer;
			e->value_layout[i] = e->prog->resources.items[inst->args[1]].layout;
			break;
		case svsl_ref_workgroup:
			result = e->workgroup_ids[inst->args[1]];
			e->value_class[i] = SpvStorageClassWorkgroup;
			break;
		case svsl_ref_const_global:
			result = e->const_global_ids[inst->args[1]];
			if (!result) eerr(e, inst->loc, "global constant initializer is not foldable");
			e->value_class[i] = SpvStorageClassPrivate;
			break;
		case svsl_ref_builtin_var:
			result = builtin_input_var(e, inst->args[1]);
			e->value_class[i] = SpvStorageClassInput;
			break;
		default:
			eerr(e, inst->loc, "bad pointer kind");
			break;
		}
		break;
	}
	case svsl_ir_chain: {
		uint32_t              base      = inst->args[0];
		const svsl_ir_inst_t *base_inst = &e->fn->insts.items[base];
		uint32_t              base_var;
		SpvStorageClass       class_;
		uint8_t               layout = e->value_layout[base];
		uint32_t              ops[18];
		uint32_t              count = 0;
		bool                  in_buffer = false;

		if (base_inst->op == svsl_ir_ptr && base_inst->args[0] == svsl_ref_buffer_member) {
			const svsl_buffer_t *buf = &e->prog->buffers.items[base_inst->args[1]];
			class_ = buf->kind == svsl_block_pushconstant ? SpvStorageClassPushConstant :
			         buf->kind == svsl_block_storagebuffer ? SpvStorageClassStorageBuffer :
			         SpvStorageClassUniform;
			base_var  = e->buffer_vars[base_inst->args[1]];
			layout    = (uint8_t)buf->layout;
			in_buffer = true;
			ops[count++] = base_var;
			ops[count++] = spv_int_const(e, (int32_t)base_inst->args[2]);
		} else if (base_inst->op == svsl_ir_ptr && base_inst->args[0] == svsl_ref_resource) {
			const svsl_resource_t *res = &e->prog->resources.items[base_inst->args[1]];
			class_    = SpvStorageClassStorageBuffer;
			base_var  = e->resource_vars[base_inst->args[1]];
			layout    = res->layout;
			in_buffer = true;
			ops[count++] = base_var;
			if (res->buffer_index < 0) // object form wraps in struct { T @data[]; }
				ops[count++] = spv_int_const(e, 0);
		} else {
			int32_t sp, sm;
			if (sroa_member_chain(e, (uint32_t)i, &sp, &sm) && e->param_sroa[sp]) {
				// scalar-replaced input: this member IS its stage-input variable.
				uint32_t mvar = e->param_member_var[sp * SVSL_EMIT_MAX_MEMBERS + sm];
				e->value_class[i]  = SpvStorageClassInput;
				e->value_layout[i] = 0;
				if (inst->aux_count == 1) {
					result = mvar; // whole member: the following load reads it directly
					break;
				}
				// sub-member (input.member.comp): access-chain into the input
				// variable with the trailing indices; the load reads the scalar.
				ops[count++] = mvar;
				for (uint32_t k = 1; k < inst->aux_count && count < 17; k++)
					ops[count++] = e->value_ids[e->fn->aux.items[inst->aux + k]];
				result = emit_value_inst(e, fs, SpvOpAccessChain,
				                         spv_ptr_type(e, SpvStorageClassInput, type_id), ops, count);
				break;
			}
			if (base == e->output_sroa_var) {
				// scalar-replaced output: this member IS its stage-output variable,
				// so the following store writes straight to the interface (analysis
				// guarantees a constant single-index member chain here).
				uint32_t m         = e->fn->insts.items[e->fn->aux.items[inst->aux]].args[0];
				result             = e->output_vars[m];
				e->value_class[i]  = SpvStorageClassOutput;
				e->value_layout[i] = 0;
				break;
			}
			base_var  = e->value_ids[base];
			class_    = (SpvStorageClass)e->value_class[base];
			in_buffer = class_ == SpvStorageClassUniform || class_ == SpvStorageClassStorageBuffer ||
			            class_ == SpvStorageClassPushConstant;
			ops[count++] = base_var;
		}
		for (uint32_t k = 0; k < inst->aux_count && count < 17; k++)
			ops[count++] = e->value_ids[e->fn->aux.items[inst->aux + k]];
		uint32_t pointee = in_buffer ? spv_type_laid(e, inst->type, (svsl_layout_)layout) : type_id;
		uint32_t ptr = spv_ptr_type(e, class_, pointee);
		result = emit_value_inst(e, fs, SpvOpAccessChain, ptr, ops, count);
		e->value_class[i]  = class_;
		e->value_layout[i] = layout;
		break;
	}
	case svsl_ir_load: {
		uint32_t ptr = e->value_ids[inst->args[0]];
		SpvStorageClass pc = (SpvStorageClass)e->value_class[inst->args[0]];
		bool in_buffer = pc == SpvStorageClassUniform || pc == SpvStorageClassStorageBuffer ||
		                 pc == SpvStorageClassPushConstant;
		if (in_buffer) {
			// buffer memory uses laid types (offset-decorated composites, bool as
			// uint): when they differ, load the laid value and rebuild the value type
			uint32_t laid_t = spv_type_laid(e, inst->type, (svsl_layout_)e->value_layout[inst->args[0]]);
			if (laid_t != type_id) {
				uint32_t laid_v = emit_value_inst(e, fs, SpvOpLoad, laid_t, (uint32_t[]){ ptr }, 1);
				result = convert_composite(e, laid_v, inst->type,
				                           (svsl_layout_)e->value_layout[inst->args[0]], true);
				break;
			}
		}
		result = emit_value_inst(e, fs, SpvOpLoad, type_id, (uint32_t[]){ ptr }, 1);
		break;
	}
	case svsl_ir_store: {
		uint32_t ptr = e->value_ids[inst->args[0]];
		SpvStorageClass pc = (SpvStorageClass)e->value_class[inst->args[0]];
		bool in_buffer = pc == SpvStorageClassUniform || pc == SpvStorageClassStorageBuffer ||
		                 pc == SpvStorageClassPushConstant;
		svsl_type_id_t vt2 = e->fn->insts.items[inst->args[1]].type;
		uint32_t value = e->value_ids[inst->args[1]];
		if (in_buffer) // laid <- value conversion; no-op when the types are shared
			value = convert_composite(e, value, vt2,
			                          (svsl_layout_)e->value_layout[inst->args[0]], false);
		svsl_spv_inst2(spv, fs, SpvOpStore, ptr, value);
		break;
	}
	case svsl_ir_construct: {
		uint32_t  stack_ops[64];
		uint32_t *ops = inst->aux_count <= 64 ? stack_ops
		              : svsl_arena_alloc(e->arena, (size_t)inst->aux_count * 4);
		for (uint32_t k = 0; k < inst->aux_count; k++)
			ops[k] = e->value_ids[e->fn->aux.items[inst->aux + k]];
		result = emit_value_inst(e, fs, SpvOpCompositeConstruct, type_id, ops, inst->aux_count);
		break;
	}
	case svsl_ir_extract:
		result = emit_value_inst(e, fs, SpvOpCompositeExtract, type_id,
		                         (uint32_t[]){ e->value_ids[inst->args[0]], inst->args[1] }, 2);
		break;
	case svsl_ir_insert:
		result = emit_value_inst(e, fs, SpvOpCompositeInsert, type_id,
		                         (uint32_t[]){ e->value_ids[inst->args[2]],
		                                       e->value_ids[inst->args[0]], inst->args[1] }, 3);
		break;
	case svsl_ir_extract_dynamic: // dynamic component index: no spill var / access chain
		result = emit_value_inst(e, fs, SpvOpVectorExtractDynamic, type_id,
		                         (uint32_t[]){ e->value_ids[inst->args[0]], e->value_ids[inst->args[1]] }, 2);
		break;
	case svsl_ir_shuffle: {
		uint32_t ops[10] = { e->value_ids[inst->args[0]], e->value_ids[inst->args[0]] };
		for (uint32_t k = 0; k < inst->args[2]; k++)
			ops[2 + k] = (inst->args[1] >> (k * 4)) & 0xF;
		result = emit_value_inst(e, fs, SpvOpVectorShuffle, type_id, ops, 2 + inst->args[2]);
		break;
	}
	case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul: case svsl_ir_div: case svsl_ir_rem: {
		const svsl_type_t *t = svsl_type_get(&e->prog->types, inst->type);
		uint32_t a = e->value_ids[inst->args[0]];
		uint32_t v = e->value_ids[inst->args[1]];

		if (t->kind == svsl_type_matrix) {
			const svsl_type_t *ta = svsl_type_get(&e->prog->types,
			                                      e->fn->insts.items[inst->args[0]].type);
			const svsl_type_t *tb = svsl_type_get(&e->prog->types,
			                                      e->fn->insts.items[inst->args[1]].type);
			if (op == svsl_ir_mul && (ta->kind == svsl_type_scalar || tb->kind == svsl_type_scalar)) {
				uint32_t mat = ta->kind == svsl_type_matrix ? a : v;
				uint32_t sca = ta->kind == svsl_type_matrix ? v : a;
				result = emit_value_inst(e, fs, SpvOpMatrixTimesScalar, type_id,
				                         (uint32_t[]){ mat, sca }, 2);
				break;
			}
			// componentwise matrix arithmetic: per-row decompose
			uint32_t scalar = spv_scalar_type(e, t->scalar);
			uint32_t row_t  = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ scalar, t->cols }, 2);
			SpvOp row_op = op == svsl_ir_add ? SpvOpFAdd : op == svsl_ir_sub ? SpvOpFSub :
			               op == svsl_ir_mul ? SpvOpFMul : SpvOpFDiv;
			uint32_t rows[4];
			for (int32_t r = 0; r < t->rows; r++) {
				uint32_t ra = emit_value_inst(e, fs, SpvOpCompositeExtract, row_t, (uint32_t[]){ a, (uint32_t)r }, 2);
				uint32_t rb = emit_value_inst(e, fs, SpvOpCompositeExtract, row_t, (uint32_t[]){ v, (uint32_t)r }, 2);
				rows[r] = emit_value_inst(e, fs, row_op, row_t, (uint32_t[]){ ra, rb }, 2);
			}
			result = emit_value_inst(e, fs, SpvOpCompositeConstruct, type_id, rows, (uint32_t)t->rows);
			break;
		}

		// float vector × scalar (peephole strips the splat so one operand is a
		// scalar) → OpVectorTimesScalar instead of splat + component-wise FMul
		if (t->kind == svsl_type_vector && op == svsl_ir_mul && scalar_is_floaty(t->scalar)) {
			const svsl_type_t *ta = svsl_type_get(&e->prog->types, e->fn->insts.items[inst->args[0]].type);
			const svsl_type_t *tb = svsl_type_get(&e->prog->types, e->fn->insts.items[inst->args[1]].type);
			if (ta->kind == svsl_type_scalar || tb->kind == svsl_type_scalar) {
				uint32_t vec = ta->kind == svsl_type_vector ? a : v;
				uint32_t sca = ta->kind == svsl_type_vector ? v : a;
				result = emit_value_inst(e, fs, SpvOpVectorTimesScalar, type_id,
				                         (uint32_t[]){ vec, sca }, 2);
				break;
			}
		}

		bool floaty = scalar_is_floaty(t->scalar);
		bool sgn    = scalar_is_signed(t->scalar);
		SpvOp o;
		switch (op) {
		case svsl_ir_add: o = floaty ? SpvOpFAdd : SpvOpIAdd; break;
		case svsl_ir_sub: o = floaty ? SpvOpFSub : SpvOpISub; break;
		case svsl_ir_mul: o = floaty ? SpvOpFMul : SpvOpIMul; break;
		case svsl_ir_div: o = floaty ? SpvOpFDiv : (sgn ? SpvOpSDiv : SpvOpUDiv); break;
		default:          o = floaty ? SpvOpFRem : (sgn ? SpvOpSRem : SpvOpUMod); break;
		}
		result = emit_op_or_spec(e, fs, i, o, type_id, inst->args, 2);
		break;
	}
	case svsl_ir_neg: {
		bool floaty = scalar_is_floaty(svsl_type_get(&e->prog->types, inst->type)->scalar);
		result = emit_op_or_spec(e, fs, i, floaty ? SpvOpFNegate : SpvOpSNegate, type_id,
		                         inst->args, 1);
		break;
	}
	case svsl_ir_bit_not:
		result = emit_op_or_spec(e, fs, i, SpvOpNot, type_id, inst->args, 1);
		break;
	case svsl_ir_log_not:
		result = emit_op_or_spec(e, fs, i, SpvOpLogicalNot, type_id, inst->args, 1);
		break;
	case svsl_ir_bit_and: case svsl_ir_bit_or: case svsl_ir_bit_xor:
	case svsl_ir_shl: case svsl_ir_shr: {
		bool sgn = scalar_is_signed(svsl_type_get(&e->prog->types, inst->type)->scalar);
		SpvOp o = op == svsl_ir_bit_and ? SpvOpBitwiseAnd :
		          op == svsl_ir_bit_or  ? SpvOpBitwiseOr :
		          op == svsl_ir_bit_xor ? SpvOpBitwiseXor :
		          op == svsl_ir_shl     ? SpvOpShiftLeftLogical :
		          sgn ? SpvOpShiftRightArithmetic : SpvOpShiftRightLogical;
		result = emit_op_or_spec(e, fs, i, o, type_id, inst->args, 2);
		break;
	}
	case svsl_ir_eq: case svsl_ir_ne: case svsl_ir_lt: case svsl_ir_le:
	case svsl_ir_gt: case svsl_ir_ge: {
		svsl_scalar_ family = value_scalar(e, inst->args[0]);
		bool floaty = scalar_is_floaty(family);
		bool sgn    = scalar_is_signed(family);
		bool booly  = family == svsl_scalar_bool;
		SpvOp o;
		switch (op) {
		case svsl_ir_eq: o = booly ? SpvOpLogicalEqual : floaty ? SpvOpFOrdEqual : SpvOpIEqual; break;
		case svsl_ir_ne: o = booly ? SpvOpLogicalNotEqual : floaty ? SpvOpFUnordNotEqual : SpvOpINotEqual; break;
		case svsl_ir_lt: o = floaty ? SpvOpFOrdLessThan : sgn ? SpvOpSLessThan : SpvOpULessThan; break;
		case svsl_ir_le: o = floaty ? SpvOpFOrdLessThanEqual : sgn ? SpvOpSLessThanEqual : SpvOpULessThanEqual; break;
		case svsl_ir_gt: o = floaty ? SpvOpFOrdGreaterThan : sgn ? SpvOpSGreaterThan : SpvOpUGreaterThan; break;
		default:         o = floaty ? SpvOpFOrdGreaterThanEqual : sgn ? SpvOpSGreaterThanEqual : SpvOpUGreaterThanEqual; break;
		}
		result = emit_op_or_spec(e, fs, i, o, type_id, inst->args, 2);
		break;
	}
	case svsl_ir_log_and: case svsl_ir_log_or:
		result = emit_op_or_spec(e, fs, i, op == svsl_ir_log_and ? SpvOpLogicalAnd : SpvOpLogicalOr,
		                         type_id, inst->args, 2);
		break;
	case svsl_ir_select: {
		uint32_t cond = e->value_ids[inst->args[0]];
		const svsl_type_t *rt2 = svsl_type_get(&e->prog->types, inst->type);
		if (rt2->kind == svsl_type_matrix) { // OpSelect can't take matrices pre-1.4
			uint32_t scalar = spv_scalar_type(e, rt2->scalar);
			uint32_t row_t  = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ scalar, rt2->cols }, 2);
			uint32_t boolt  = svsl_spv_type(spv, SpvOpTypeBool, NULL, 0);
			uint32_t bvec   = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ boolt, rt2->cols }, 2);
			uint32_t csplat[4] = { cond, cond, cond, cond };
			uint32_t cvec = emit_value_inst(e, fs, SpvOpCompositeConstruct, bvec, csplat, (uint32_t)rt2->cols);
			uint32_t rows[4];
			for (int32_t r = 0; r < rt2->rows; r++) {
				uint32_t ra = emit_value_inst(e, fs, SpvOpCompositeExtract, row_t,
				                              (uint32_t[]){ e->value_ids[inst->args[1]], (uint32_t)r }, 2);
				uint32_t rb = emit_value_inst(e, fs, SpvOpCompositeExtract, row_t,
				                              (uint32_t[]){ e->value_ids[inst->args[2]], (uint32_t)r }, 2);
				rows[r] = emit_value_inst(e, fs, SpvOpSelect, row_t, (uint32_t[]){ cvec, ra, rb }, 3);
			}
			result = emit_value_inst(e, fs, SpvOpCompositeConstruct, type_id, rows, (uint32_t)rt2->rows);
			break;
		}
		const svsl_type_t *ct  = svsl_type_get(&e->prog->types,
		                                       e->fn->insts.items[inst->args[0]].type);
		if (rt2->kind == svsl_type_vector && ct->kind == svsl_type_scalar) {
			// pre-1.4 OpSelect needs the condition splatted to the result shape
			uint32_t boolt = svsl_spv_type(spv, SpvOpTypeBool, NULL, 0);
			uint32_t vt = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ boolt, rt2->count }, 2);
			uint32_t ops[6] = { cond, cond, cond, cond };
			cond = emit_value_inst(e, fs, SpvOpCompositeConstruct, vt, ops, (uint32_t)rt2->count);
		}
		if (cond == e->value_ids[inst->args[0]]) { // scalar: may stay a spec expression
			result = emit_op_or_spec(e, fs, i, SpvOpSelect, type_id, inst->args, 3);
		} else {
			result = emit_value_inst(e, fs, SpvOpSelect, type_id,
			                         (uint32_t[]){ cond,
			                                       e->value_ids[inst->args[1]],
			                                       e->value_ids[inst->args[2]] }, 3);
		}
		break;
	}
	case svsl_ir_convert: {
		svsl_type_id_t from_t = e->fn->insts.items[inst->args[0]].type;
		uint32_t       value  = e->value_ids[inst->args[0]];
		if (spv_type_for(e, from_t) == type_id) { result = value; break; } // half ↔ float
		svsl_scalar_ from = svsl_type_get(&e->prog->types, from_t)->scalar;
		svsl_scalar_ to   = svsl_type_get(&e->prog->types, inst->type)->scalar;
		bool ff = scalar_is_floaty(from), tf = scalar_is_floaty(to);
		bool fs_ = scalar_is_signed(from), ts = scalar_is_signed(to);

		if (from == svsl_scalar_bool) { // bool → number: select(b, 1, 0)
			uint32_t one  = tf ? float_splat(e, inst->type, 1.0f) : const_splat(e, inst->type, 1);
			uint32_t zero = tf ? float_splat(e, inst->type, 0.0f) : const_splat(e, inst->type, 0);
			result = emit_value_inst(e, fs, SpvOpSelect, type_id, (uint32_t[]){ value, one, zero }, 3);
			break;
		}
		if (to == svsl_scalar_bool) { // number → bool: value != 0
			uint32_t zero = ff ? float_splat(e, from_t, 0.0f) : const_splat(e, from_t, 0);
			result = emit_value_inst(e, fs, ff ? SpvOpFUnordNotEqual : SpvOpINotEqual, type_id,
			                         (uint32_t[]){ value, zero }, 2);
			break;
		}
		SpvOp o;
		if      (ff && tf)  o = SpvOpFConvert;
		else if (ff && ts)  o = SpvOpConvertFToS;
		else if (ff)        o = SpvOpConvertFToU;
		else if (tf && fs_) o = SpvOpConvertSToF;
		else if (tf)        o = SpvOpConvertUToF;
		else if (svsl_scalar_size(from) == svsl_scalar_size(to)) o = SpvOpBitcast;
		else {
			// integer resize. Widening must extend by the SOURCE signedness
			// (zero- vs sign-extend); narrowing truncates either way, so it
			// follows the result. The validator requires the op and the result
			// signedness to agree, so mismatches resize within the source's
			// signedness first, then bitcast to the final type.
			bool op_signed = svsl_scalar_size(to) < svsl_scalar_size(from) ? ts : fs_;
			o = op_signed ? SpvOpSConvert : SpvOpUConvert;
			if (op_signed != ts) {
				const svsl_type_t *tt  = svsl_type_get(&e->prog->types, inst->type);
				svsl_scalar_       mid =
					svsl_scalar_size(to) == 1 ? (op_signed ? svsl_scalar_int8  : svsl_scalar_uint8) :
					svsl_scalar_size(to) == 2 ? (op_signed ? svsl_scalar_int16 : svsl_scalar_uint16) :
					svsl_scalar_size(to) == 8 ? (op_signed ? svsl_scalar_int64 : svsl_scalar_uint64) :
					                            (op_signed ? svsl_scalar_int32 : svsl_scalar_uint32);
				svsl_types_t  *types    = (svsl_types_t *)&e->prog->types;
				svsl_type_id_t mid_type = tt->kind == svsl_type_vector
					? svsl_type_vector_id(types, mid, tt->count)
					: svsl_type_scalar_id(types, mid);
				value = emit_value_inst(e, fs, o, spv_type_for(e, mid_type),
				                        (uint32_t[]){ value }, 1);
				o = SpvOpBitcast;
			}
		}
		result = emit_value_inst(e, fs, o, type_id, (uint32_t[]){ value }, 1);
		break;
	}
	case svsl_ir_mat_mul: {
		const svsl_type_t *ta = svsl_type_get(&e->prog->types, e->fn->insts.items[inst->args[0]].type);
		const svsl_type_t *tb = svsl_type_get(&e->prog->types, e->fn->insts.items[inst->args[1]].type);
		uint32_t a = e->value_ids[inst->args[0]];
		uint32_t v = e->value_ids[inst->args[1]];
		// HLSL mul with row-representation matrices: swap into GLSL operand order
		if (ta->kind == svsl_type_vector)       // mul(v, M) = M_spv * v
			result = emit_value_inst(e, fs, SpvOpMatrixTimesVector, type_id, (uint32_t[]){ v, a }, 2);
		else if (tb->kind == svsl_type_vector)  // mul(M, v) = v * M_spv
			result = emit_value_inst(e, fs, SpvOpVectorTimesMatrix, type_id, (uint32_t[]){ v, a }, 2);
		else                                     // mul(A, B) = B_spv * A_spv
			result = emit_value_inst(e, fs, SpvOpMatrixTimesMatrix, type_id, (uint32_t[]){ v, a }, 2);
		break;
	}
	case svsl_ir_intrinsic:
		result = emit_intrinsic(e, i, inst);
		break;
	case svsl_ir_tex: {
		int32_t  tex     = (int32_t)inst->args[0];
		uint32_t sampler = inst->args[1];
		int32_t  method  = (int32_t)(inst->args[2] & 0xFF);
		int32_t  channel = (int32_t)(int8_t)((inst->args[2] >> 8) & 0xFF);
		const svsl_resource_t *res = &e->prog->resources.items[tex];
		const svsl_type_t     *ot  = svsl_type_get(&e->prog->types, res->type);

		uint32_t a0 = inst->aux_count > 0 ? e->value_ids[e->fn->aux.items[inst->aux + 0]] : 0;
		uint32_t a1 = inst->aux_count > 1 ? e->value_ids[e->fn->aux.items[inst->aux + 1]] : 0;
		uint32_t a2 = inst->aux_count > 2 ? e->value_ids[e->fn->aux.items[inst->aux + 2]] : 0;

		// structured Buffer<T> never becomes an image op
		if (ot->kind == svsl_type_buffer) {
			if (method == svsl_method_get_dimensions) { // count of the runtime array (member 0)
				result = emit_value_inst(e, fs, SpvOpArrayLength, type_id,
				                         (uint32_t[]){ e->resource_vars[tex], 0 }, 2);
				break;
			}
			uint32_t ptr  = spv_ptr_type(e, SpvStorageClassStorageBuffer, type_id);
			uint32_t zero = spv_int_const(e, 0);
			result = emit_value_inst(e, fs, SpvOpAccessChain, ptr,
			                         (uint32_t[]){ e->resource_vars[tex], zero, a0 }, 3);
			result = emit_value_inst(e, fs, SpvOpLoad, type_id, (uint32_t[]){ result }, 1);
			break;
		}
		if (ot->kind == svsl_type_subpass) { // SubpassInput[MS].Load([sample])
			uint32_t img = emit_value_inst(e, fs, SpvOpLoad, e->resource_img_type[tex],
			                               (uint32_t[]){ e->resource_vars[tex] }, 1);
			// Vulkan requires the SubpassData coordinate to be a CONSTANT (0,0)
			svsl_types_t *tt = (svsl_types_t *)&e->prog->types;
			uint32_t c0  = const_splat(e, svsl_type_vector_id(tt, svsl_scalar_int32, 2), 0);
			const svsl_type_t *se = svsl_type_get(tt, ot->elem);
			uint32_t sc  = spv_scalar_type(e, se->scalar);
			uint32_t v4  = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ sc, 4 }, 2);
			uint32_t ops[4] = { img, c0, SpvImageOperandsSampleMask, a0 };
			uint32_t raw = emit_value_inst(e, fs, SpvOpImageRead, v4, ops,
			                               ot->multisampled ? 4u : 2u);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		if (ot->kind == svsl_type_tileimage) { // TileImage.Load()
			svsl_spv_cap(spv, SpvCapabilityTileImageColorReadAccessEXT);
			svsl_spv_extension(spv, "SPV_EXT_shader_tile_image");
			uint32_t img = emit_value_inst(e, fs, SpvOpLoad, e->resource_img_type[tex],
			                               (uint32_t[]){ e->resource_vars[tex] }, 1);
			const svsl_type_t *te = svsl_type_get(&e->prog->types, ot->elem);
			uint32_t sc = spv_scalar_type(e, te->scalar);
			uint32_t v4 = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ sc, 4 }, 2);
			uint32_t raw = emit_value_inst(e, fs, SpvOpColorAttachmentReadEXT, v4,
			                               (uint32_t[]){ img }, 1);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}

		const svsl_type_t *elem = svsl_type_get(&e->prog->types, ot->elem);
		uint32_t scalar_id = spv_scalar_type(e, elem->scalar);
		uint32_t vec4_id   = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ scalar_id, 4 }, 2);

		// ordinary texturing marks its resources so QCOM-exclusive uses conflict
		if (method != svsl_method_sample_weighted && method != svsl_method_box_filter &&
		    method != svsl_method_block_match) {
			qcom_classify(e, tex, svsl_qcom_use_plain, inst->loc);
			if (sampler != SVSL_IR_NONE) qcom_classify(e, (int32_t)sampler, svsl_qcom_use_plain, inst->loc);
		}

		switch ((svsl_method_)method) {
		case svsl_method_sample:
		case svsl_method_sample_bias: {
			uint32_t si = load_sampled_image(e, tex, sampler);
			uint32_t ops[5] = { si, a0 };
			uint32_t count  = 2;
			SpvOp    sample_op = SpvOpImageSampleImplicitLod;
			if (e->fn->entry->stage != svsl_stage_pixel) {
				// implicit-lod sampling only exists in fragment shaders; other
				// stages sample the top mip (bias has no meaning there either)
				sample_op    = SpvOpImageSampleExplicitLod;
				ops[count++] = SpvImageOperandsLodMask;
				ops[count++] = spv_float_const(e, 0.0f);
			} else if (method == svsl_method_sample_bias) {
				ops[count++] = SpvImageOperandsBiasMask;
				ops[count++] = a1;
			}
			uint32_t raw = emit_value_inst(e, fs, sample_op, vec4_id, ops, count);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		case svsl_method_sample_level: {
			uint32_t si = load_sampled_image(e, tex, sampler);
			uint32_t ops[4] = { si, a0, SpvImageOperandsLodMask, a1 };
			uint32_t raw = emit_value_inst(e, fs, SpvOpImageSampleExplicitLod, vec4_id, ops, 4);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		case svsl_method_sample_grad: {
			uint32_t si = load_sampled_image(e, tex, sampler);
			uint32_t ops[5] = { si, a0, SpvImageOperandsGradMask, a1, a2 };
			uint32_t raw = emit_value_inst(e, fs, SpvOpImageSampleExplicitLod, vec4_id, ops, 5);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		case svsl_method_sample_cmp: {
			uint32_t si = load_sampled_image(e, tex, sampler);
			result = emit_value_inst(e, fs, SpvOpImageSampleDrefImplicitLod,
			                         spv_scalar_type(e, svsl_scalar_float32),
			                         (uint32_t[]){ si, a0, a1 }, 3);
			break;
		}
		case svsl_method_sample_cmp_level_zero: {
			uint32_t si   = load_sampled_image(e, tex, sampler);
			uint32_t zero = spv_float_const(e, 0.0f);
			uint32_t ops[5] = { si, a0, a1, SpvImageOperandsLodMask, zero };
			result = emit_value_inst(e, fs, SpvOpImageSampleDrefExplicitLod,
			                         spv_scalar_type(e, svsl_scalar_float32), ops, 5);
			break;
		}
		case svsl_method_gather: {
			uint32_t si = load_sampled_image(e, tex, sampler);
			if (channel == 4) { // GatherCmp
				result = emit_value_inst(e, fs, SpvOpImageDrefGather, vec4_id,
				                         (uint32_t[]){ si, a0, a1 }, 3);
			} else {
				uint32_t comp = spv_int_const(e, channel < 0 ? 0 : channel);
				result = emit_value_inst(e, fs, SpvOpImageGather, vec4_id,
				                         (uint32_t[]){ si, a0, comp }, 3);
			}
			result = shrink_to(e, result, 4, inst->type);
			break;
		}
		case svsl_method_load: { // fetch: last coord component is the mip
			uint32_t si  = load_sampled_image(e, tex, SVSL_IR_NONE);
			uint32_t img = res->sampler_slot >= 0
			             ? emit_value_inst(e, fs, SpvOpImage, e->resource_img_type[tex], &si, 1)
			             : si;
			if (ot->multisampled) { // Texture2DMS.Load(coord, sample); no mips
				uint32_t ops[4] = { img, a0, SpvImageOperandsSampleMask, a1 };
				uint32_t raw = emit_value_inst(e, fs, SpvOpImageFetch, vec4_id, ops, 4);
				result = shrink_to(e, raw, 4, inst->type);
				break;
			}
			int32_t n = ot->dim == svsl_texdim_1d ? 1 : ot->dim == svsl_texdim_3d ? 3 : 2;
			n += ot->arrayed ? 1 : 0;
			uint32_t i32 = spv_scalar_type(e, svsl_scalar_int32);
			uint32_t coord, mip;
			if (n == 1) {
				coord = emit_value_inst(e, fs, SpvOpCompositeExtract, i32, (uint32_t[]){ a0, 0 }, 2);
				mip   = emit_value_inst(e, fs, SpvOpCompositeExtract, i32, (uint32_t[]){ a0, 1 }, 2);
			} else {
				uint32_t vt = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ i32, (uint32_t)n }, 2);
				uint32_t ops[8] = { a0, a0 };
				for (int32_t k = 0; k < n; k++) ops[2 + k] = (uint32_t)k;
				coord = emit_value_inst(e, fs, SpvOpVectorShuffle, vt, ops, 2 + (uint32_t)n);
				mip   = emit_value_inst(e, fs, SpvOpCompositeExtract, i32, (uint32_t[]){ a0, (uint32_t)n }, 2);
			}
			uint32_t ops[4] = { img, coord, SpvImageOperandsLodMask, mip };
			uint32_t raw = emit_value_inst(e, fs, SpvOpImageFetch, vec4_id, ops, 4);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		case svsl_method_get_dimensions: {
			svsl_spv_cap(spv, SpvCapabilityImageQuery);
			uint32_t img;
			if (ot->kind == svsl_type_texture) {
				uint32_t si = load_sampled_image(e, tex, SVSL_IR_NONE);
				img = res->sampler_slot >= 0
				    ? emit_value_inst(e, fs, SpvOpImage, e->resource_img_type[tex], &si, 1)
				    : si;
				if (channel == 1) { // trailing level/sample-count output
					result = emit_value_inst(e, fs, ot->multisampled ? SpvOpImageQuerySamples
					                                                 : SpvOpImageQueryLevels,
					                         type_id, &img, 1);
					break;
				}
				if (ot->multisampled) { // MS images have no mips: plain size query
					result = emit_value_inst(e, fs, SpvOpImageQuerySize, type_id,
					                         (uint32_t[]){ img }, 1);
					break;
				}
				uint32_t lod = inst->aux_count > 0 ? a0 : spv_int_const(e, 0);
				result = emit_value_inst(e, fs, SpvOpImageQuerySizeLod, type_id,
				                         (uint32_t[]){ img, lod }, 2);
			} else {
				img = emit_value_inst(e, fs, SpvOpLoad, e->resource_img_type[tex],
				                      (uint32_t[]){ e->resource_vars[tex] }, 1);
				result = emit_value_inst(e, fs, SpvOpImageQuerySize, type_id, (uint32_t[]){ img }, 1);
			}
			break;
		}
		case svsl_method_query_lod: { // CalculateLevelOfDetail(sampler, coords)
			svsl_spv_cap(spv, SpvCapabilityImageQuery);
			uint32_t si  = load_sampled_image(e, tex, sampler);
			uint32_t f32 = spv_scalar_type(e, svsl_scalar_float32);
			uint32_t v2  = svsl_spv_type(spv, SpvOpTypeVector, (uint32_t[]){ f32, 2 }, 2);
			uint32_t lods = emit_value_inst(e, fs, SpvOpImageQueryLod, v2, (uint32_t[]){ si, a0 }, 2);
			result = emit_value_inst(e, fs, SpvOpCompositeExtract, type_id,
			                         (uint32_t[]){ lods, channel == 1 ? 1u : 0u }, 2);
			break;
		}
		case svsl_method_sample_weighted: { // QCOM: MxN weighted average
			svsl_spv_cap(spv, SpvCapabilityTextureSampleWeightedQCOM);
			svsl_spv_extension(spv, "SPV_QCOM_image_processing");
			svsl_spv_require_version(spv, 0x00010400); // extension's SPIR-V floor
			int32_t  weight = (int32_t)inst->args[3];
			uint32_t si     = load_sampled_image(e, tex, sampler);
			uint32_t wi     = load_sampled_image(e, weight, sampler);
			qcom_classify(e, tex, svsl_qcom_use_plain, inst->loc);
			qcom_classify(e, weight, svsl_qcom_use_weight, inst->loc);
			qcom_classify(e, (int32_t)sampler, svsl_qcom_use_ip_sampler, inst->loc);
			qcom_decorate_once(e, e->resource_vars[weight], SpvDecorationWeightTextureQCOM);
			uint32_t raw = emit_value_inst(e, fs, SpvOpImageSampleWeightedQCOM, vec4_id,
			                               (uint32_t[]){ si, a0, wi }, 3);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		case svsl_method_box_filter: { // QCOM: filter over an arbitrary box size
			svsl_spv_cap(spv, SpvCapabilityTextureBoxFilterQCOM);
			svsl_spv_extension(spv, "SPV_QCOM_image_processing");
			svsl_spv_require_version(spv, 0x00010400);
			uint32_t si = load_sampled_image(e, tex, sampler);
			qcom_classify(e, tex, svsl_qcom_use_plain, inst->loc);
			qcom_classify(e, (int32_t)sampler, svsl_qcom_use_ip_sampler, inst->loc);
			uint32_t raw = emit_value_inst(e, fs, SpvOpImageBoxFilterQCOM, vec4_id,
			                               (uint32_t[]){ si, a0, a1 }, 3);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		case svsl_method_block_match: { // QCOM: SAD/SSD block error metric
			static const SpvOp bm_ops[6] = { // svsl_block_match_ order
				SpvOpImageBlockMatchSADQCOM,       SpvOpImageBlockMatchSSDQCOM,
				SpvOpImageBlockMatchWindowSADQCOM, SpvOpImageBlockMatchWindowSSDQCOM,
				SpvOpImageBlockMatchGatherSADQCOM, SpvOpImageBlockMatchGatherSSDQCOM,
			};
			bool window = channel == svsl_block_match_window_sad ||
			              channel == svsl_block_match_window_ssd;
			svsl_spv_cap(spv, SpvCapabilityTextureBlockMatchQCOM);
			svsl_spv_extension(spv, "SPV_QCOM_image_processing");
			svsl_spv_require_version(spv, 0x00010400);
			if (channel >= svsl_block_match_window_sad) { // Window/Gather forms
				svsl_spv_cap(spv, SpvCapabilityTextureBlockMatch2QCOM);
				svsl_spv_extension(spv, "SPV_QCOM_image_processing2");
			}
			int32_t  ref = (int32_t)inst->args[3];
			uint32_t ti, ri;
			if (window) { // classifies + decorates inside (combined-var handling)
				ti = qcom_window_pair(e, tex, sampler, inst->loc);
				ri = qcom_window_pair(e, ref, sampler, inst->loc);
			} else {
				ti = load_sampled_image(e, tex, sampler);
				ri = load_sampled_image(e, ref, sampler);
				qcom_classify(e, tex, svsl_qcom_use_block_match, inst->loc);
				qcom_classify(e, ref, svsl_qcom_use_block_match, inst->loc);
				qcom_classify(e, (int32_t)sampler, svsl_qcom_use_ip_sampler, inst->loc);
				qcom_decorate_once(e, e->resource_vars[tex], SpvDecorationBlockMatchTextureQCOM);
				qcom_decorate_once(e, e->resource_vars[ref], SpvDecorationBlockMatchTextureQCOM);
			}
			uint32_t raw = emit_value_inst(e, fs, bm_ops[channel], vec4_id,
			                               (uint32_t[]){ ti, a0, ri, a1, a2 }, 5);
			result = shrink_to(e, raw, 4, inst->type);
			break;
		}
		default:
			eerr(e, inst->loc, "texture method has no SPIR-V mapping yet");
			break;
		}
		break;
	}
	case svsl_ir_image_load: {
		int32_t tex = (int32_t)inst->args[0];
		const svsl_resource_t *res = &e->prog->resources.items[tex];
		const svsl_type_t *ot = svsl_type_get(&e->prog->types, res->type);
		if (svsl_image_format_for(&e->prog->types, ot) == SpvImageFormatUnknown)
			svsl_spv_cap(spv, SpvCapabilityStorageImageReadWithoutFormat);
		uint32_t img = emit_value_inst(e, fs, SpvOpLoad, e->resource_img_type[tex],
		                               (uint32_t[]){ e->resource_vars[tex] }, 1);
		const svsl_type_t *elem = svsl_type_get(&e->prog->types, ot->elem);
		uint32_t vec4 = svsl_spv_type(spv, SpvOpTypeVector,
		                              (uint32_t[]){ spv_scalar_type(e, elem->scalar), 4 }, 2);
		uint32_t raw = emit_value_inst(e, fs, SpvOpImageRead, vec4,
		                               (uint32_t[]){ img, e->value_ids[inst->args[1]] }, 2);
		result = shrink_to(e, raw, 4, inst->type);
		break;
	}
	case svsl_ir_image_store: {
		int32_t tex = (int32_t)inst->args[0];
		const svsl_resource_t *res = &e->prog->resources.items[tex];
		const svsl_type_t *ot = svsl_type_get(&e->prog->types, res->type);
		if (svsl_image_format_for(&e->prog->types, ot) == SpvImageFormatUnknown)
			svsl_spv_cap(spv, SpvCapabilityStorageImageWriteWithoutFormat);
		uint32_t img = emit_value_inst(e, fs, SpvOpLoad, e->resource_img_type[tex],
		                               (uint32_t[]){ e->resource_vars[tex] }, 1);
		svsl_spv_inst3(spv, fs, SpvOpImageWrite, img, e->value_ids[inst->args[1]],
		               e->value_ids[inst->args[2]]);
		break;
	}
	case svsl_ir_image_atomic:
	case svsl_ir_atomic: {
		uint32_t ptr;
		if (op == svsl_ir_image_atomic) {
			int32_t tex = (int32_t)inst->args[0];
			uint32_t ptr_t = spv_ptr_type(e, SpvStorageClassImage, type_id);
			ptr = emit_value_inst(e, fs, SpvOpImageTexelPointer, ptr_t,
			                      (uint32_t[]){ e->resource_vars[tex],
			                                    e->value_ids[inst->args[1]],
			                                    spv_uint_const(e, 0) }, 3);
		} else {
			ptr = e->value_ids[inst->args[0]];
		}
		uint32_t atomic_op = inst->args[3] & 0xFF;
		uint32_t order     = inst->args[3] >> 8; // svsl_mem_order_ (0 relaxed)

		// scope: inferred from the destination's storage — a groupshared atomic
		// synchronizes only within the workgroup, so Workgroup scope suffices;
		// buffer (and image) atomics are device-visible.
		SpvStorageClass pc = op == svsl_ir_image_atomic ? SpvStorageClassImage
		                                                : (SpvStorageClass)e->value_class[inst->args[0]];
		uint32_t scope = spv_uint_const(e, pc == SpvStorageClassWorkgroup ? SpvScopeWorkgroup
		                                                                  : SpvScopeDevice);
		// semantics: relaxed (0) stays None — bit-identical to glslang. A requested
		// order carries its ordering bits OR the storage-class memory bit.
		uint32_t sem_bits = 0;
		if (order > 0) {
			static const uint32_t ord_bits[] = {
				0, SpvMemorySemanticsAcquireMask, SpvMemorySemanticsReleaseMask,
				SpvMemorySemanticsAcquireReleaseMask, SpvMemorySemanticsSequentiallyConsistentMask };
			uint32_t storage = op == svsl_ir_image_atomic ? SpvMemorySemanticsImageMemoryMask :
			                   pc == SpvStorageClassWorkgroup ? SpvMemorySemanticsWorkgroupMemoryMask :
			                                                    SpvMemorySemanticsUniformMemoryMask;
			sem_bits = ord_bits[order] | storage;
		}
		uint32_t sem = spv_uint_const(e, sem_bits);

		bool sgn = scalar_is_signed(svsl_type_get(&e->prog->types, inst->type)->scalar);
		uint32_t value = op == svsl_ir_image_atomic ? e->value_ids[inst->args[2]]
		                                            : e->value_ids[inst->args[1]];
		if (atomic_op == 8) { // compare_exchange(dest, compare, value)
			uint32_t comparator = e->value_ids[inst->args[1]];
			uint32_t new_value  = e->value_ids[inst->args[2]];
			// a failed compare-exchange performs no store, so its "unequal"
			// semantics must not release (SPIR-V requires this)
			uint32_t uneq = sem_bits & ~(uint32_t)SpvMemorySemanticsReleaseMask;
			if (uneq & SpvMemorySemanticsAcquireReleaseMask)
				uneq = (uneq & ~(uint32_t)SpvMemorySemanticsAcquireReleaseMask) |
				       SpvMemorySemanticsAcquireMask;
			result = emit_value_inst(e, fs, SpvOpAtomicCompareExchange, type_id,
			                         (uint32_t[]){ ptr, scope, sem, spv_uint_const(e, uneq),
			                                       new_value, comparator }, 6);
			break;
		}
		static const SpvOp signed_ops[] = { SpvOpAtomicIAdd, SpvOpAtomicISub, SpvOpAtomicSMin,
		                                    SpvOpAtomicSMax, SpvOpAtomicAnd, SpvOpAtomicOr,
		                                    SpvOpAtomicXor, SpvOpAtomicExchange };
		static const SpvOp unsigned_ops[] = { SpvOpAtomicIAdd, SpvOpAtomicISub, SpvOpAtomicUMin,
		                                      SpvOpAtomicUMax, SpvOpAtomicAnd, SpvOpAtomicOr,
		                                      SpvOpAtomicXor, SpvOpAtomicExchange };
		SpvOp o = (sgn ? signed_ops : unsigned_ops)[atomic_op < 8 ? atomic_op : 0];
		if (scalar_is_floaty(svsl_type_get(&e->prog->types, inst->type)->scalar)) {
			// SPV_EXT_shader_atomic_float: sema restricts float destinations to
			// add/min/max/exchange (exchange needs no capability of its own)
			if (atomic_op == 0) { // add
				svsl_spv_cap(spv, SpvCapabilityAtomicFloat32AddEXT);
				svsl_spv_extension(spv, "SPV_EXT_shader_atomic_float_add");
				o = SpvOpAtomicFAddEXT;
			} else if (atomic_op == 2 || atomic_op == 3) { // min / max
				svsl_spv_cap(spv, SpvCapabilityAtomicFloat32MinMaxEXT);
				svsl_spv_extension(spv, "SPV_EXT_shader_atomic_float_min_max");
				o = atomic_op == 2 ? SpvOpAtomicFMinEXT : SpvOpAtomicFMaxEXT;
			}
		}
		result = emit_value_inst(e, fs, o, type_id, (uint32_t[]){ ptr, scope, sem, value }, 4);
		break;
	}
	case svsl_ir_spirv_asm:
		result = emit_spirv_asm(e, inst);
		break;
	case svsl_ir_bitfield_extract: {
		bool s = scalar_is_signed(svsl_type_get(&e->prog->types, inst->type)->scalar);
		result = emit_value_inst(e, fs, s ? SpvOpBitFieldSExtract : SpvOpBitFieldUExtract, type_id,
		                         (uint32_t[]){ e->value_ids[inst->args[0]], e->value_ids[inst->args[1]],
		                                       e->value_ids[inst->args[2]] }, 3);
		break;
	}
	case svsl_ir_bitfield_insert:
		result = emit_value_inst(e, fs, SpvOpBitFieldInsert, type_id,
		                         (uint32_t[]){ e->value_ids[inst->args[0]], e->value_ids[inst->args[1]],
		                                       e->value_ids[inst->args[2]], e->value_ids[inst->args[3]] }, 4);
		break;
	default:
		break;
	}

	e->value_ids[i] = result;
	if (result && op != svsl_ir_const && op != svsl_ir_spec_const)
		relax(e, result, inst->type);
	if (result && (inst->flags & svsl_ir_flag_precise)) // `precise`: forbid fma contraction
		svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, result, SpvDecorationNoContraction);
}
