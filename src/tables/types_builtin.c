#include "types_builtin.h"
#include "../sema/types.h"

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

bool svsl_type_name_is_resource(svsl_str_t name) {
	for (int32_t i = 0; i < (int32_t)(sizeof(resource_names) / sizeof(resource_names[0])); i++)
		if (svsl_str_eq_cstr(name, resource_names[i])) return true;
	return false;
}

bool svsl_type_name_is_builtin(svsl_str_t name) {
	if (svsl_str_eq_cstr(name, "void")) return true;
	svsl_scalar_ scalar;
	int32_t      rows, cols;
	return svsl_scalar_name_parse(name, &scalar, &rows, &cols) || svsl_type_name_is_resource(name);
}
