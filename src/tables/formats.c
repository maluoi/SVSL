#include "formats.h"

#include "../../vendor/spirv.h"

typedef struct format_row_t {
	const char *name;
	uint32_t    format;
} format_row_t;

static const format_row_t rows[] = {
	{ "rgba32f", SpvImageFormatRgba32f },  { "rgba16f", SpvImageFormatRgba16f },
	{ "rg32f", SpvImageFormatRg32f },      { "rg16f", SpvImageFormatRg16f },
	{ "r32f", SpvImageFormatR32f },        { "r16f", SpvImageFormatR16f },
	{ "rgba8", SpvImageFormatRgba8 },      { "rgba8_snorm", SpvImageFormatRgba8Snorm },
	{ "r11g11b10f", SpvImageFormatR11fG11fB10f },
	{ "rgba16", SpvImageFormatRgba16 },    { "rgb10a2", SpvImageFormatRgb10A2 },
	{ "rg16", SpvImageFormatRg16 },        { "rg8", SpvImageFormatRg8 },
	{ "r16", SpvImageFormatR16 },          { "r8", SpvImageFormatR8 },
	{ "rgba32i", SpvImageFormatRgba32i },  { "rgba16i", SpvImageFormatRgba16i },
	{ "rgba8i", SpvImageFormatRgba8i },    { "r32i", SpvImageFormatR32i },
	{ "rgba32u", SpvImageFormatRgba32ui }, { "rgba16u", SpvImageFormatRgba16ui },
	{ "rgba8u", SpvImageFormatRgba8ui },   { "r32u", SpvImageFormatR32ui },
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
