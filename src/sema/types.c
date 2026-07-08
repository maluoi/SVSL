#include "types.h"

#include <stdio.h>
#include <string.h>

static bool type_eq(const svsl_type_t *a, const svsl_type_t *b) {
	return a->kind == b->kind && a->scalar == b->scalar && a->count == b->count &&
	       a->rows == b->rows && a->cols == b->cols && a->elem == b->elem &&
	       a->array_count == b->array_count && a->struct_index == b->struct_index &&
	       a->dim == b->dim && a->arrayed == b->arrayed &&
	       a->multisampled == b->multisampled &&
	       a->is_comparison == b->is_comparison && a->is_rw == b->is_rw &&
	       svsl_str_eq(a->format, b->format);
}

// IEEE 754 binary16 encoding with round-to-nearest-even (float16 constants)
uint32_t svsl_f32_to_f16_bits(float f) {
	uint32_t x;
	memcpy(&x, &f, 4);
	uint32_t sign = (x >> 16) & 0x8000;
	int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
	uint32_t man  = x & 0x7FFFFF;
	if (((x >> 23) & 0xFF) == 0xFF) return sign | 0x7C00 | (man ? 0x200 : 0); // inf/nan
	if (exp >= 31) return sign | 0x7C00;                                      // overflow → inf
	if (exp <= 0) {                                                           // subnormal/zero
		if (exp < -10) return sign;
		man |= 0x800000;
		uint32_t shift    = (uint32_t)(14 - exp);
		uint32_t half     = man >> shift;
		uint32_t rest     = man & ((1u << shift) - 1u);   // discarded bits
		uint32_t half_bit = 1u << (shift - 1);            // the tie point
		if (rest > half_bit || (rest == half_bit && (half & 1))) half++; // round half to even
		return sign | half;
	}
	uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
	uint32_t rest = man & 0x1FFF;                                     // discarded bits
	// round half to even (tie → keep the even LSB); carry ripples into the exponent correctly
	if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) half++;
	return half;
}

svsl_type_id_t svsl_type_intern(svsl_types_t *types, svsl_type_t type) {
	// --half=strict16 re-types every half at the single creation point, so no
	// half type can exist anywhere downstream (declarations, literals, casts)
	if (types->half_strict16 && type.scalar == svsl_scalar_half &&
	    (type.kind == svsl_type_scalar || type.kind == svsl_type_vector ||
	     type.kind == svsl_type_matrix))
		type.scalar = svsl_scalar_float16;
	for (int32_t i = 0; i < types->types.count; i++)
		if (type_eq(&types->types.items[i], &type)) return i;
	svsl_array_push(types->arena, &types->types, type);
	return types->types.count - 1;
}

svsl_type_id_t svsl_type_scalar_id(svsl_types_t *types, svsl_scalar_ scalar) {
	return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_scalar, .scalar = scalar });
}
svsl_type_id_t svsl_type_vector_id(svsl_types_t *types, svsl_scalar_ scalar, int32_t count) {
	return svsl_type_intern(types, (svsl_type_t){
		.kind = svsl_type_vector, .scalar = scalar, .count = (uint8_t)count });
}
svsl_type_id_t svsl_type_matrix_id(svsl_types_t *types, svsl_scalar_ scalar, int32_t rows, int32_t cols) {
	return svsl_type_intern(types, (svsl_type_t){
		.kind = svsl_type_matrix, .scalar = scalar, .rows = (uint8_t)rows, .cols = (uint8_t)cols });
}
svsl_type_id_t svsl_type_array_id(svsl_types_t *types, svsl_type_id_t elem, int32_t count) {
	return svsl_type_intern(types, (svsl_type_t){
		.kind = svsl_type_array, .elem = elem, .array_count = count });
}

const svsl_type_t *svsl_type_get(const svsl_types_t *types, svsl_type_id_t id) {
	// error recovery routinely leaves SVSL_TYPE_NONE behind; give it a real
	// object so downstream switches fall through instead of reading items[-1]
	static const svsl_type_t none = { .kind = svsl_type_void };
	if (id < 0 || id >= types->types.count) return &none;
	return &types->types.items[id];
}

// --- scalar names ---------------------------------------------------------------

typedef struct scalar_name_row_t {
	const char  *name;
	svsl_scalar_ scalar;
} scalar_name_row_t;

// longest-match-first so "int8" wins over "int"+"8"
static const scalar_name_row_t scalar_names[] = {
	{ "min16float", svsl_scalar_half    },
	{ "float16",    svsl_scalar_float16 },
	{ "float32",    svsl_scalar_float32 },
	{ "float64",    svsl_scalar_float64 },
	{ "int8",       svsl_scalar_int8    },
	{ "int16",      svsl_scalar_int16   },
	{ "int32",      svsl_scalar_int32   },
	{ "int64",      svsl_scalar_int64   },
	{ "uint8",      svsl_scalar_uint8   },
	{ "uint16",     svsl_scalar_uint16  },
	{ "uint32",     svsl_scalar_uint32  },
	{ "uint64",     svsl_scalar_uint64  },
	{ "double",     svsl_scalar_float64 },
	{ "float",      svsl_scalar_float32 },
	{ "half",       svsl_scalar_half    },
	{ "bool",       svsl_scalar_bool    },
	{ "uint",       svsl_scalar_uint32  },
	{ "int",        svsl_scalar_int32   },
};

bool svsl_scalar_name_parse(svsl_str_t name, svsl_scalar_ *out_scalar,
                            int32_t *out_rows, int32_t *out_cols) {
	for (int32_t i = 0; i < (int32_t)(sizeof(scalar_names) / sizeof(scalar_names[0])); i++) {
		const char *scalar = scalar_names[i].name;
		int32_t     len    = (int32_t)strlen(scalar);
		if (name.len < len || memcmp(name.ptr, scalar, (size_t)len) != 0) continue;

		svsl_str_t rest = svsl_str_slice(name, len, name.len);
		if (rest.len == 0) {
			*out_scalar = scalar_names[i].scalar;
			*out_rows   = 0;
			*out_cols   = 0;
			return true;
		}
		if (rest.len == 1 && rest.ptr[0] >= '1' && rest.ptr[0] <= '4') {
			*out_scalar = scalar_names[i].scalar;
			*out_rows   = rest.ptr[0] - '0';
			*out_cols   = 0;
			return true;
		}
		if (rest.len == 3 && rest.ptr[0] >= '1' && rest.ptr[0] <= '4' &&
		    rest.ptr[1] == 'x' && rest.ptr[2] >= '1' && rest.ptr[2] <= '4') {
			*out_scalar = scalar_names[i].scalar;
			*out_rows   = rest.ptr[0] - '0';
			*out_cols   = rest.ptr[2] - '0';
			return true;
		}
	}
	return false;
}

int32_t svsl_scalar_size(svsl_scalar_ scalar) {
	switch (scalar) {
	case svsl_scalar_bool:    return 4; // 32-bit in memory interfaces
	case svsl_scalar_int8:   case svsl_scalar_uint8:   return 1;
	case svsl_scalar_int16:  case svsl_scalar_uint16:
	case svsl_scalar_float16:                          return 2;
	case svsl_scalar_int64:  case svsl_scalar_uint64:
	case svsl_scalar_float64:                          return 8;
	case svsl_scalar_half: // RelaxedPrecision float32: layouts never depend on fp16 support
	default:                                           return 4;
	}
}

bool svsl_type_is_resource(const svsl_type_t *t) {
	return t->kind == svsl_type_texture || t->kind == svsl_type_sampler ||
	       t->kind == svsl_type_image || t->kind == svsl_type_buffer ||
	       t->kind == svsl_type_subpass || t->kind == svsl_type_tileimage;
}

static const char *scalar_display_name(svsl_scalar_ scalar) {
	switch (scalar) {
	case svsl_scalar_bool:    return "bool";
	case svsl_scalar_int8:    return "int8";
	case svsl_scalar_int16:   return "int16";
	case svsl_scalar_int32:   return "int";
	case svsl_scalar_int64:   return "int64";
	case svsl_scalar_uint8:   return "uint8";
	case svsl_scalar_uint16:  return "uint16";
	case svsl_scalar_uint32:  return "uint";
	case svsl_scalar_uint64:  return "uint64";
	case svsl_scalar_float16: return "float16";
	case svsl_scalar_float32: return "float";
	case svsl_scalar_float64: return "double";
	case svsl_scalar_half:    return "half";
	default:                  return "?";
	}
}

const char *svsl_type_name(const svsl_types_t *types, svsl_type_id_t id) {
	char buf[128];
	if (id == SVSL_TYPE_NONE) return "<error>";
	const svsl_type_t *t = svsl_type_get(types, id);

	switch (t->kind) {
	case svsl_type_void:   return "void";
	case svsl_type_scalar: return scalar_display_name(t->scalar);
	case svsl_type_vector:
		snprintf(buf, sizeof(buf), "%s%d", scalar_display_name(t->scalar), t->count);
		break;
	case svsl_type_matrix:
		snprintf(buf, sizeof(buf), "%s%dx%d", scalar_display_name(t->scalar), t->rows, t->cols);
		break;
	case svsl_type_array:
		if (t->array_count > 0)
			snprintf(buf, sizeof(buf), "%s[%d]", svsl_type_name(types, t->elem), t->array_count);
		else
			snprintf(buf, sizeof(buf), "%s[]", svsl_type_name(types, t->elem));
		break;
	case svsl_type_struct:
		return types->structs.items[t->struct_index].name.ptr; // interned NUL-terminated
	case svsl_type_texture: {
		static const char *dims[] = { "1D", "2D", "3D", "Cube" };
		snprintf(buf, sizeof(buf), "Texture%s%s%s<%s>", dims[t->dim],
		         t->multisampled ? "MS" : "", t->arrayed ? "Array" : "",
		         svsl_type_name(types, t->elem));
		break;
	}
	case svsl_type_image: {
		static const char *dims[] = { "1D", "2D", "3D", "Cube" };
		snprintf(buf, sizeof(buf), "Image%s%s<%s%s%.*s>", dims[t->dim], t->arrayed ? "Array" : "",
		         svsl_type_name(types, t->elem), t->format.len ? "," : "",
		         t->format.len, t->format.ptr ? t->format.ptr : "");
		break;
	}
	case svsl_type_sampler: return t->is_comparison ? "SamplerComparison" : "Sampler";
	case svsl_type_subpass:
		snprintf(buf, sizeof(buf), "SubpassInput%s<%s>", t->multisampled ? "MS" : "",
		         svsl_type_name(types, t->elem));
		break;
	case svsl_type_tileimage:
		snprintf(buf, sizeof(buf), "TileImage<%s>", svsl_type_name(types, t->elem));
		break;
	case svsl_type_buffer:
		snprintf(buf, sizeof(buf), "%sStructuredBuffer<%s>", t->is_rw ? "RW" : "",
		         svsl_type_name(types, t->elem));
		break;
	}
	return svsl_arena_strndup(types->arena, buf, strlen(buf));
}
