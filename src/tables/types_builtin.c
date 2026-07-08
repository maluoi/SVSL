#include "types_builtin.h"

#include <string.h>

// Scalar base type names. The trailing vector/matrix suffix is validated
// separately (only 1-4 / NxN), so order here doesn't matter — "int8" can't
// mis-parse as "int" + "8".
static const char *scalar_names[] = {
	"min16float", // legacy alias of half
	"float16", "float32", "float64",
	"int8", "int16", "int32", "int64",
	"uint8", "uint16", "uint32", "uint64",
	"double", "float", "half", "bool", "uint", "int",
};

static const char *resource_names[] = {
	"Texture1D", "Texture2D", "Texture3D", "TextureCube",
	"Texture1DArray", "Texture2DArray", "TextureCubeArray",
	"RWTexture1D", "RWTexture2D", "RWTexture3D",
	"RWTexture1DArray", "RWTexture2DArray",
	"Image1D", "Image2D", "Image3D", "ImageCube",
	"Image1DArray", "Image2DArray", "ImageCubeArray",
	"Texture2DMS", "SubpassInput", "SubpassInputMS", "TileImage",
	"Buffer", "StructuredBuffer", "RWStructuredBuffer",
	"SamplerState", "SamplerComparisonState",
	"Sampler", "SamplerComparison",
};

// name = scalar [1-4] | scalar [1-4]x[1-4] | scalar
static bool is_scalar_vector_matrix(svsl_str_t name) {
	for (int32_t i = 0; i < (int32_t)(sizeof(scalar_names) / sizeof(scalar_names[0])); i++) {
		const char *scalar = scalar_names[i];
		int32_t     len    = (int32_t)strlen(scalar);
		if (name.len < len || memcmp(name.ptr, scalar, (size_t)len) != 0) continue;

		svsl_str_t rest = svsl_str_slice(name, len, name.len);
		if (rest.len == 0) return true;                                     // float
		if (rest.len == 1 && rest.ptr[0] >= '1' && rest.ptr[0] <= '4') return true; // float4
		if (rest.len == 3 && rest.ptr[0] >= '1' && rest.ptr[0] <= '4' &&
		    rest.ptr[1] == 'x' && rest.ptr[2] >= '1' && rest.ptr[2] <= '4') return true; // float4x4
	}
	return false;
}

bool svsl_type_name_is_resource(svsl_str_t name) {
	for (int32_t i = 0; i < (int32_t)(sizeof(resource_names) / sizeof(resource_names[0])); i++)
		if (svsl_str_eq_cstr(name, resource_names[i])) return true;
	return false;
}

bool svsl_type_name_is_builtin(svsl_str_t name) {
	if (svsl_str_eq_cstr(name, "void")) return true;
	return is_scalar_vector_matrix(name) || svsl_type_name_is_resource(name);
}
