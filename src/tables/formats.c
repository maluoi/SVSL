#include "formats.h"

#include "../../vendor/spirv.h"

typedef struct format_row_t {
	const char *name;
	uint32_t    format;
} format_row_t;

// Names are glslang's canonical layout-format strings (getLayoutFormatString),
// the exact spellings skshaderc accepts in [[vk::image_format("...")]]. r64ui/
// r64i are omitted: they need SpvCapabilityInt64ImageEXT, which emit doesn't
// yet produce, and accepting the name without the capability would silently
// emit invalid SPIR-V.
static const format_row_t rows[] = {
	// float
	{ "rgba32f", SpvImageFormatRgba32f },       { "rgba16f", SpvImageFormatRgba16f },
	{ "rg32f", SpvImageFormatRg32f },           { "rg16f", SpvImageFormatRg16f },
	{ "r11f_g11f_b10f", SpvImageFormatR11fG11fB10f },
	{ "r32f", SpvImageFormatR32f },             { "r16f", SpvImageFormatR16f },
	// unorm
	{ "rgba16", SpvImageFormatRgba16 },         { "rgb10_a2", SpvImageFormatRgb10A2 },
	{ "rgba8", SpvImageFormatRgba8 },           { "rg16", SpvImageFormatRg16 },
	{ "rg8", SpvImageFormatRg8 },               { "r16", SpvImageFormatR16 },
	{ "r8", SpvImageFormatR8 },
	// snorm
	{ "rgba16_snorm", SpvImageFormatRgba16Snorm }, { "rgba8_snorm", SpvImageFormatRgba8Snorm },
	{ "rg16_snorm", SpvImageFormatRg16Snorm },  { "rg8_snorm", SpvImageFormatRg8Snorm },
	{ "r16_snorm", SpvImageFormatR16Snorm },    { "r8_snorm", SpvImageFormatR8Snorm },
	// signed int
	{ "rgba32i", SpvImageFormatRgba32i },       { "rgba16i", SpvImageFormatRgba16i },
	{ "rgba8i", SpvImageFormatRgba8i },         { "rg32i", SpvImageFormatRg32i },
	{ "rg16i", SpvImageFormatRg16i },           { "rg8i", SpvImageFormatRg8i },
	{ "r32i", SpvImageFormatR32i },             { "r16i", SpvImageFormatR16i },
	{ "r8i", SpvImageFormatR8i },
	// unsigned int
	{ "rgba32ui", SpvImageFormatRgba32ui },     { "rgba16ui", SpvImageFormatRgba16ui },
	{ "rgba8ui", SpvImageFormatRgba8ui },       { "rg32ui", SpvImageFormatRg32ui },
	{ "rg16ui", SpvImageFormatRg16ui },         { "rgb10_a2ui", SpvImageFormatRgb10a2ui },
	{ "rg8ui", SpvImageFormatRg8ui },           { "r32ui", SpvImageFormatR32ui },
	{ "r16ui", SpvImageFormatR16ui },           { "r8ui", SpvImageFormatR8ui },
};

bool svsl_image_format_find(svsl_str_t name, uint32_t *out_spv) {
	for (int32_t i = 0; i < (int32_t)(sizeof(rows) / sizeof(rows[0])); i++)
		if (svsl_str_eq_cstr(name, rows[i].name)) {
			if (out_spv) *out_spv = rows[i].format;
			return true;
		}
	return false;
}

bool svsl_image_format_extended(uint32_t spv_format) {
	switch (spv_format) {
	case SpvImageFormatRg32f:       case SpvImageFormatRg16f:    case SpvImageFormatR11fG11fB10f:
	case SpvImageFormatR16f:        case SpvImageFormatRgba16:   case SpvImageFormatRgb10A2:
	case SpvImageFormatRg16:        case SpvImageFormatRg8:      case SpvImageFormatR16:
	case SpvImageFormatR8:          case SpvImageFormatRgba16Snorm:
	case SpvImageFormatRg16Snorm:   case SpvImageFormatRg8Snorm: case SpvImageFormatR16Snorm:
	case SpvImageFormatR8Snorm:     case SpvImageFormatRg32i:    case SpvImageFormatRg16i:
	case SpvImageFormatRg8i:        case SpvImageFormatR16i:     case SpvImageFormatR8i:
	case SpvImageFormatRgb10a2ui:   case SpvImageFormatRg32ui:   case SpvImageFormatRg16ui:
	case SpvImageFormatRg8ui:       case SpvImageFormatR16ui:    case SpvImageFormatR8ui:
		return true;
	default:
		return false;
	}
}

#include "../sema/types.h"

uint32_t svsl_image_format_for(const svsl_types_t *types, const svsl_type_t *t) {
	if (t->format.len > 0) {
		uint32_t format = SpvImageFormatUnknown;
		svsl_image_format_find(t->format, &format);
		return format;
	}
	const svsl_type_t *elem  = svsl_type_get(types, t->elem);
	int32_t            count = elem->kind == svsl_type_vector ? elem->count : 1;
	switch (elem->scalar) {
	case svsl_scalar_int32:
		return count == 1 ? SpvImageFormatR32i  : count == 2 ? SpvImageFormatRg32i  : SpvImageFormatRgba32i;
	case svsl_scalar_uint32:
		return count == 1 ? SpvImageFormatR32ui : count == 2 ? SpvImageFormatRg32ui : SpvImageFormatRgba32ui;
	default:
		return count == 1 ? SpvImageFormatR32f  : count == 2 ? SpvImageFormatRg32f  : SpvImageFormatRgba32f;
	}
}
