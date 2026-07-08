#include "sks_write.h"

#include <svsl/svsl.h> // SVSL_SKS_VERSION (the reader validates against the same constant)
#include "../ir/usage.h"
#include "../tables/formats.h"
#include "../tables/semantics.h"
#include "../util/array.h"
#include "../../vendor/spirv.h"

#include <string.h>

typedef svsl_array_t(uint8_t) byte_buf_t;

typedef struct sks_t {
	svsl_arena_t *arena;
	byte_buf_t    out;
} sks_t;

static void wbytes(sks_t *w, const void *data, size_t size) {
	const uint8_t *bytes = data;
	for (size_t i = 0; i < size; i++)
		svsl_array_push(w->arena, &w->out, bytes[i]);
}
static void wu8 (sks_t *w, uint8_t v)  { wbytes(w, &v, 1); }
static void wu16(sks_t *w, uint16_t v) { wbytes(w, &v, 2); }
static void wu32(sks_t *w, uint32_t v) { wbytes(w, &v, 4); }
static void wi32(sks_t *w, int32_t v)  { wbytes(w, &v, 4); }
static void wu64(sks_t *w, uint64_t v) { wbytes(w, &v, 8); }

// zero-padded fixed-size string
static void wstr(sks_t *w, svsl_str_t s, int32_t size) {
	for (int32_t i = 0; i < size; i++)
		wu8(w, i < s.len ? (uint8_t)s.ptr[i] : 0);
}

// skr_bind_t: { uint16 slot; uint8 stage_bits; uint8 register_type }
static void wbind(sks_t *w, uint16_t slot, uint8_t stage_bits, uint8_t register_type) {
	wu16(w, slot);
	wu8(w, stage_bits);
	wu8(w, register_type);
}

// skr_register_ values (sksc_file.h)
enum { reg_default = 0, reg_vertex, reg_index, reg_constant, reg_texture,
       reg_read_buffer, reg_readwrite, reg_readwrite_tex, reg_input_attachment };
// sksc_shader_var_ values
enum { var_none = 0, var_int, var_uint, var_uint8, var_float, var_double };
// skr_vertex_fmt_ values
enum { fmt_none = 0, fmt_f64, fmt_f32, fmt_f16, fmt_i32, fmt_i16, fmt_i8,
       fmt_i32n, fmt_i16n, fmt_i8n, fmt_ui32, fmt_ui16, fmt_ui8, fmt_ui32n,
       fmt_ui16n, fmt_ui8n };
// skr_semantic_ values
enum { sem_none = 0, sem_position, sem_texcoord, sem_normal, sem_binormal,
       sem_tangent, sem_color, sem_psize, sem_blendweight, sem_blendindices };

// the SPIR-V descriptor binding (b+0, t/s+100, u+200; direct form raw)
static uint16_t sks_binding(const svsl_binding_t *bind) {
	if (bind->direct) return (uint16_t)bind->slot;
	switch (bind->cls) {
	case 't': case 's': return (uint16_t)(bind->slot + 100);
	case 'u':           return (uint16_t)(bind->slot + 200);
	default:            return (uint16_t)bind->slot; // buffers keep the register number
	}
}

static uint16_t var_type_of(const svsl_types_t *types, svsl_type_id_t id, uint16_t *out_count) {
	const svsl_type_t *t     = svsl_type_get(types, id);
	int32_t            elems = 1;
	if (t->kind == svsl_type_array) {
		elems = t->array_count;
		t     = svsl_type_get(types, t->elem);
	}
	int32_t components = t->kind == svsl_type_vector ? t->count :
	                     t->kind == svsl_type_matrix ? t->rows * t->cols : 1;
	*out_count = (uint16_t)(components * elems);
	if (t->kind == svsl_type_struct) return var_none; // struct members: type_name carries the info
	switch (t->scalar) {
	case svsl_scalar_bool:                             return var_uint; // stored as uint 0/1
	case svsl_scalar_int8: case svsl_scalar_int16:
	case svsl_scalar_int32: case svsl_scalar_int64:    return var_int;
	case svsl_scalar_uint8:                            return var_uint8;
	case svsl_scalar_uint16: case svsl_scalar_uint32:
	case svsl_scalar_uint64:                           return var_uint;
	case svsl_scalar_float64:                          return var_double;
	default:                                           return var_float;
	}
}

// base type name without array suffix ("float4x4", struct names as written)
static svsl_str_t var_type_name(const svsl_types_t *types, svsl_type_id_t id) {
	const svsl_type_t *t = svsl_type_get(types, id);
	if (t->kind == svsl_type_array) return var_type_name(types, t->elem);
	return svsl_str(svsl_type_name(types, id));
}

static int32_t semantic_of(svsl_str_t semantic, uint8_t *out_slot) {
	struct row { const char *name; int32_t sem; };
	static const struct row rows[] = {
		{ "SV_POSITION",  sem_position },  { "POSITION", sem_position },
		{ "TEXCOORD",     sem_texcoord },  { "NORMAL",   sem_normal },
		{ "BINORMAL",     sem_binormal },  { "TANGENT",  sem_tangent },
		{ "COLOR",        sem_color },     { "PSIZE",    sem_psize },
		{ "BLENDWEIGHT",  sem_blendweight },
		{ "BLENDINDICES", sem_blendindices },
	};
	*out_slot = 0;
	for (int32_t i = 0; i < (int32_t)(sizeof(rows) / sizeof(rows[0])); i++) {
		const char *name = rows[i].name;
		int32_t     len  = (int32_t)strlen(name);
		if (semantic.len < len) continue;
		bool match = true;
		for (int32_t k = 0; k < len && match; k++) {
			char c = semantic.ptr[k];
			if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
			if (c != name[k]) match = false;
		}
		if (!match) continue;
		int32_t slot = 0;
		bool    digits_ok = true;
		for (int32_t k = len; k < semantic.len; k++) {
			if (semantic.ptr[k] < '0' || semantic.ptr[k] > '9') { digits_ok = false; break; }
			slot = slot * 10 + (semantic.ptr[k] - '0');
		}
		if (!digits_ok) continue;
		*out_slot = (uint8_t)slot;
		return rows[i].sem;
	}
	return sem_none;
}

static int32_t vertex_format_of(const svsl_types_t *types, svsl_type_id_t id, uint8_t *out_count) {
	const svsl_type_t *t = svsl_type_get(types, id);
	*out_count = t->kind == svsl_type_vector ? t->count : 1;
	switch (t->scalar) {
	case svsl_scalar_float64: return fmt_f64;
	case svsl_scalar_float16: return fmt_f16;
	case svsl_scalar_int32:   return fmt_i32;
	case svsl_scalar_int16:   return fmt_i16;
	case svsl_scalar_int8:    return fmt_i8;
	case svsl_scalar_uint32:  return fmt_ui32;
	case svsl_scalar_uint16:  return fmt_ui16;
	case svsl_scalar_uint8:   return fmt_ui8;
	default:                  return fmt_f32; // float32 and half
	}
}

// --- device-feature mask ------------------------------------------------------------
//
// Each bit answers one device-level feature question the runtime must settle
// before pipeline creation. A bit can cover several capabilities plus their
// extension. Anything the SPIR-V declares that maps to no bit sets bit 63, so
// a runtime never silently under-checks a newer compiler's output. The blob's
// own OpCapability/OpExtension lists stay the exhaustive ground truth.
enum {
	sks_feat_float16          = 0,  // Float16 arithmetic
	sks_feat_storage16        = 1,  // 16-bit buffer access
	sks_feat_storage8         = 2,  // 8-bit buffer access (+SPV_KHR_8bit_storage)
	sks_feat_extended_formats = 3,  // StorageImageExtendedFormats
	sks_feat_image_atomics    = 4,  // OpImageTexelPointer (needs STORAGE_IMAGE_ATOMIC formats)
	sks_feat_subgroups        = 5,  // GroupNonUniform family
	sks_feat_wave_size        = 6,  // VK_EXT_subgroup_size_control
	sks_feat_multiview        = 7,
	sks_feat_demote           = 8,  // (+SPV_EXT_demote_to_helper_invocation)
	sks_feat_int64            = 9,
	sks_feat_float64          = 10,
	sks_feat_int16            = 11,
	sks_feat_int8             = 12,
	sks_feat_formatless       = 13, // StorageImageRead/WriteWithoutFormat
	sks_feat_tile_image       = 14, // (+SPV_EXT_shader_tile_image)
	sks_feat_float_atomics    = 15, // (+SPV_EXT_shader_atomic_float[2])
	sks_feat_unknown          = 63, // capability/extension with no assigned bit
};

typedef struct feat_row_t { uint32_t cap; uint8_t bit; } feat_row_t;
static const feat_row_t feature_caps[] = {
	{ SpvCapabilityFloat16,                          sks_feat_float16 },
	{ SpvCapabilityStorageBuffer16BitAccess,         sks_feat_storage16 },
	{ SpvCapabilityUniformAndStorageBuffer16BitAccess, sks_feat_storage16 },
	{ SpvCapabilityStoragePushConstant16,            sks_feat_storage16 },
	{ SpvCapabilityStorageBuffer8BitAccess,          sks_feat_storage8 },
	{ SpvCapabilityUniformAndStorageBuffer8BitAccess,  sks_feat_storage8 },
	{ SpvCapabilityStoragePushConstant8,             sks_feat_storage8 },
	{ SpvCapabilityStorageImageExtendedFormats,      sks_feat_extended_formats },
	{ SpvCapabilityGroupNonUniform,                  sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformVote,              sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformBallot,            sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformArithmetic,        sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformShuffle,           sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformShuffleRelative,   sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformClustered,         sks_feat_subgroups },
	{ SpvCapabilityGroupNonUniformQuad,              sks_feat_subgroups },
	{ SpvCapabilityMultiView,                        sks_feat_multiview },
	{ SpvCapabilityDemoteToHelperInvocationEXT,      sks_feat_demote },
	{ SpvCapabilityInt64,                            sks_feat_int64 },
	{ SpvCapabilityFloat64,                          sks_feat_float64 },
	{ SpvCapabilityInt16,                            sks_feat_int16 },
	{ SpvCapabilityInt8,                             sks_feat_int8 },
	{ SpvCapabilityStorageImageReadWithoutFormat,    sks_feat_formatless },
	{ SpvCapabilityStorageImageWriteWithoutFormat,   sks_feat_formatless },
	{ SpvCapabilityTileImageColorReadAccessEXT,      sks_feat_tile_image },
	{ SpvCapabilityTileImageDepthReadAccessEXT,      sks_feat_tile_image },
	{ SpvCapabilityTileImageStencilReadAccessEXT,    sks_feat_tile_image },
	{ SpvCapabilityAtomicFloat32AddEXT,              sks_feat_float_atomics },
	{ SpvCapabilityAtomicFloat32MinMaxEXT,           sks_feat_float_atomics },
};
// capabilities every Vulkan 1.1 runtime satisfies — no bit, never unknown
static const uint32_t baseline_caps[] = {
	SpvCapabilityShader, SpvCapabilityImageQuery, SpvCapabilitySampled1D,
	SpvCapabilityImage1D, SpvCapabilityInputAttachment, SpvCapabilityDerivativeControl,
};
static const struct { const char *name; uint8_t bit; } feature_exts[] = {
	{ "SPV_KHR_8bit_storage",               sks_feat_storage8 },
	{ "SPV_EXT_demote_to_helper_invocation", sks_feat_demote },
	{ "SPV_EXT_shader_tile_image",          sks_feat_tile_image },
	{ "SPV_EXT_shader_atomic_float_add",    sks_feat_float_atomics },
	{ "SPV_EXT_shader_atomic_float_min_max", sks_feat_float_atomics },
};

// Writes one stage's size + SPIR-V. The runtime loads entry points by fixed
// name (sk_renderer binds "vs"/"ps"/"cs"), so whatever the author called the
// entry function, the blob's OpEntryPoint is renamed to the canonical stage
// name on the way into the container. Raw .spv output keeps the source name.
static void write_stage_spirv(sks_t *w, const svsl_spirv_blob_t *blob, svsl_stage_ stage) {
	const char *canon = stage == svsl_stage_vertex ? "vs" :
	                    stage == svsl_stage_pixel  ? "ps" : "cs";
	const uint32_t *words = blob->words;
	int32_t         at    = 5;
	while (at < blob->word_count) { // find OpEntryPoint (one entry per module)
		uint32_t word_count = words[at] >> 16;
		if (word_count == 0) break;
		if ((words[at] & 0xFFFF) == SpvOpEntryPoint) break;
		at += (int32_t)word_count;
	}
	if (at >= blob->word_count) { // no entry point: pass the blob through
		wu32(w, (uint32_t)blob->word_count * 4);
		wbytes(w, words, (size_t)blob->word_count * 4);
		return;
	}

	uint32_t    old_count = words[at] >> 16;
	const char *old_name  = (const char *)&words[at + 3];
	uint32_t    old_words = ((uint32_t)strlen(old_name) + 1 + 3) / 4;
	uint32_t    new_count = old_count - old_words + 1; // canonical names fit one word

	wu32(w, (uint32_t)(blob->word_count - (int32_t)old_words + 1) * 4);
	wbytes(w, words, (size_t)at * 4);
	wu32(w, (new_count << 16) | SpvOpEntryPoint);
	wu32(w, words[at + 1]); // execution model
	wu32(w, words[at + 2]); // function id
	wu32(w, (uint32_t)(uint8_t)canon[0] | ((uint32_t)(uint8_t)canon[1] << 8));
	wbytes(w, &words[at + 3 + old_words], // interface ids + everything after
	       (size_t)(blob->word_count - at - 3 - (int32_t)old_words) * 4);
}

// scans one SPIR-V blob's declarations (and one opcode with format-feature
// implications) and accumulates feature bits
static uint64_t feature_bits(const svsl_spirv_blob_t *blob) {
	uint64_t        bits  = 0;
	const uint32_t *words = blob->words;
	for (int32_t i = 5; i < blob->word_count; ) {
		uint32_t word_count = words[i] >> 16;
		uint32_t opcode     = words[i] & 0xFFFF;
		if (word_count == 0) break;

		if (opcode == SpvOpCapability) {
			uint32_t cap   = words[i + 1];
			bool     known = false;
			for (size_t k = 0; k < sizeof(feature_caps) / sizeof(feature_caps[0]); k++)
				if (feature_caps[k].cap == cap) { bits |= 1ull << feature_caps[k].bit; known = true; }
			for (size_t k = 0; k < sizeof(baseline_caps) / sizeof(baseline_caps[0]); k++)
				if (baseline_caps[k] == cap) known = true;
			if (!known) bits |= 1ull << sks_feat_unknown;
		} else if (opcode == SpvOpExtension) {
			const char *name  = (const char *)&words[i + 1];
			bool        known = false;
			for (size_t k = 0; k < sizeof(feature_exts) / sizeof(feature_exts[0]); k++)
				if (strncmp(name, feature_exts[k].name, (size_t)(word_count - 1) * 4) == 0) {
					bits |= 1ull << feature_exts[k].bit;
					known = true;
				}
			if (!known) bits |= 1ull << sks_feat_unknown;
		} else if (opcode == SpvOpImageTexelPointer) {
			bits |= 1ull << sks_feat_image_atomics;
		}
		i += (int32_t)word_count;
	}
	return bits;
}

// resource shape byte: bits 0-2 dimension (0 = unreported, 1 = 2D, 2 = 3D,
// 3 = cube, 4 = 1D), bit 3 arrayed, bit 4 multisampled, bit 5 comparison-sampled
static uint8_t resource_shape(const svsl_types_t *types, const svsl_resource_t *res) {
	const svsl_type_t *t = svsl_type_get(types, res->type);
	if (t->kind == svsl_type_array) t = svsl_type_get(types, t->elem);
	uint8_t shape = 0;
	if (t->kind == svsl_type_texture || t->kind == svsl_type_image) {
		shape = t->dim == svsl_texdim_3d   ? 2 :
		        t->dim == svsl_texdim_cube ? 3 :
		        t->dim == svsl_texdim_1d   ? 4 : 1;
		if (t->arrayed)      shape |= 1 << 3;
		if (t->multisampled) shape |= 1 << 4;
	}
	if (t->kind == svsl_type_subpass) {
		shape = 1; // input attachments read as 2D
		if (t->multisampled) shape |= 1 << 4;
	}
	if (t->kind == svsl_type_sampler && t->is_comparison) shape |= 1 << 5;
	return shape;
}

// sksc's op-count rule, verbatim (sksc_meta.cpp)
static void count_ops(const svsl_spirv_blob_t *blob, int32_t *out_total,
                      int32_t *out_tex, int32_t *out_flow) {
	*out_total = *out_tex = *out_flow = 0;
	const uint32_t *words = blob->words;
	for (int32_t i = 5; i < blob->word_count; ) {
		uint32_t word_count = words[i] >> 16;
		uint32_t opcode     = words[i] & 0xFFFF;
		if (word_count == 0) break;

		bool is_metadata =
			(opcode <= 8) || (opcode >= 11 && opcode <= 17) ||
			(opcode >= 19 && opcode <= 39) || (opcode >= 41 && opcode <= 52) ||
			(opcode == 59) || (opcode >= 71 && opcode <= 76);
		if (!is_metadata) {
			(*out_total)++;
			if      (opcode >= 87 && opcode <= 98)   (*out_tex)++;
			else if (opcode >= 249 && opcode <= 251) (*out_flow)++;
		}
		i += (int32_t)word_count;
	}
}

void svsl_sks_write(svsl_arena_t *arena, const svsl_program_t *prog,
                    const svsl_ir_module_t *module, const svsl_spirv_blob_t *blobs,
                    svsl_sks_blob_t *out_blob) {
	sks_t w = { .arena = arena };
	svsl_usage_t usage;
	svsl_ir_analyze_usage(arena, prog, module, &usage);

	// only uniform-kind buffers reflect in the buffer table; storage buffers are
	// resources and push constants have no SKS representation yet
	int32_t *buffer_indices = svsl_arena_alloc(arena, (size_t)(prog->buffers.count > 0 ? prog->buffers.count : 1) * sizeof(int32_t));
	int32_t buffer_count = 0;
	for (int32_t i = 0; i < prog->buffers.count; i++)
		if (prog->buffers.items[i].kind == svsl_block_uniform && usage.buffer_stages[i] != 0)
			buffer_indices[buffer_count++] = i; // unused buffers are dropped (like skshaderc)
	// sort by register slot (matches skshaderc's enumeration for the corpus)
	for (int32_t a = 0; a < buffer_count; a++)
		for (int32_t b = a + 1; b < buffer_count; b++)
			if (prog->buffers.items[buffer_indices[b]].bind.slot <
			    prog->buffers.items[buffer_indices[a]].bind.slot) {
				int32_t tmp = buffer_indices[a];
				buffer_indices[a] = buffer_indices[b];
				buffer_indices[b] = tmp;
			}

	// resources: everything except paired samplers (fused away)
	int32_t *resource_indices = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * sizeof(int32_t));
	int32_t resource_count = 0;
	for (int32_t i = 0; i < prog->resources.count; i++) {
		const svsl_resource_t *res = &prog->resources.items[i];
		if (usage.resource_stages[i] == 0) continue; // unused: dropped (like skshaderc)
		if (res->kind == svsl_res_tileimage) continue; // tile memory, nothing to bind
		if (res->kind == svsl_res_sampler) {
			bool paired = false;
			for (int32_t k = 0; k < prog->resources.count; k++)
				if (prog->resources.items[k].kind == svsl_res_texture &&
				    prog->resources.items[k].sampler_slot == res->bind.slot &&
				    prog->resources.items[k].bind.space == res->bind.space) paired = true;
			if (paired) continue;
		}
		resource_indices[resource_count++] = i;
	}

	int32_t used_inputs = 0;
	for (int32_t i = 0; i < prog->vertex_inputs.count; i++)
		if (usage.vs_input_used[i]) used_inputs++;

	const char tag[8] = { 'S', 'K', 'S', 'H', 'A', 'D', 'E', 'R' };
	wbytes(&w, tag, 8);
	wu16(&w, SVSL_SKS_VERSION); // one version at a time; runtimes refuse anything else
	wu32(&w, (uint32_t)module->func_count);
	wstr(&w, prog->name_from_meta ? prog->name : (svsl_str_t){0}, 256); // empty without //--name
	wu32(&w, (uint32_t)buffer_count);
	wu32(&w, (uint32_t)resource_count);
	wi32(&w, used_inputs);
	wu32(&w, (uint32_t)prog->spec_consts.count);
	uint64_t features = 0; // device-feature mask + reserved growth room
	for (int32_t i = 0; i < module->func_count; i++) {
		features |= feature_bits(&blobs[i]);
		if (module->funcs[i].entry->wave_size > 0)
			features |= 1ull << sks_feat_wave_size;
	}
	if (prog->wave_size > 0) features |= 1ull << sks_feat_wave_size;
	wu64(&w, features);
	wu64(&w, 0);

	int32_t ops_v[3] = {0}, ops_p[3] = {0};
	for (int32_t i = 0; i < module->func_count; i++) {
		if (module->funcs[i].entry->stage == svsl_stage_vertex)
			count_ops(&blobs[i], &ops_v[0], &ops_v[1], &ops_v[2]);
		if (module->funcs[i].entry->stage == svsl_stage_pixel)
			count_ops(&blobs[i], &ops_p[0], &ops_p[1], &ops_p[2]);
	}
	wi32(&w, ops_v[0]); wi32(&w, ops_v[1]); wi32(&w, ops_v[2]);
	wi32(&w, ops_p[0]); wi32(&w, ops_p[1]); wi32(&w, ops_p[2]);
	wu32(&w, (uint32_t)prog->wave_size);

	for (int32_t bi = 0; bi < buffer_count; bi++) {
		const svsl_buffer_t *buf = &prog->buffers.items[buffer_indices[bi]];
		wstr(&w, buf->name, 32);
		wu8(&w, (uint8_t)buf->bind.space);
		wbind(&w, sks_binding(&buf->bind), usage.buffer_stages[buffer_indices[bi]], reg_constant);
		wu32(&w, buf->size);
		wu32(&w, (uint32_t)buf->members.count);
		if (buf->defaults) {
			wu32(&w, buf->size);
			wbytes(&w, buf->defaults, buf->size);
		} else {
			wu32(&w, 0);
		}
		for (int32_t m = 0; m < buf->members.count; m++) {
			const svsl_buf_member_t *member = &buf->members.items[m];
			uint16_t type_count;
			uint16_t type = var_type_of(&prog->types, member->type, &type_count);
			wstr(&w, member->name, 32);
			wstr(&w, member->extra, 64);
			wstr(&w, var_type_name(&prog->types, member->type), 32);
			wu32(&w, member->offset);
			wu32(&w, member->size);
			wu16(&w, type);
			wu16(&w, type_count);
		}
	}

	for (int32_t i = 0; i < prog->vertex_inputs.count; i++) {
		if (!usage.vs_input_used[i]) continue;
		const svsl_vertex_input_t *in = &prog->vertex_inputs.items[i];
		uint8_t count, slot;
		int32_t format   = vertex_format_of(&prog->types, in->type, &count);
		int32_t semantic = semantic_of(in->semantic, &slot);
		wi32(&w, format);
		wu8(&w, count);
		wi32(&w, semantic);
		wu8(&w, slot);
	}

	for (int32_t ri = 0; ri < resource_count; ri++) {
		int32_t                index = resource_indices[ri];
		const svsl_resource_t *res   = &prog->resources.items[index];
		uint8_t reg =
			res->kind == svsl_res_texture       ? reg_texture :
			res->kind == svsl_res_sampler       ? reg_texture :
			res->kind == svsl_res_structured    ? reg_read_buffer :
			res->kind == svsl_res_rw_structured ? reg_readwrite :
			res->kind == svsl_res_image         ? reg_readwrite_tex : reg_input_attachment;
		wstr(&w, res->name, 32);
		wstr(&w, res->value, 64);
		wstr(&w, res->tags, 64);
		wbind(&w, sks_binding(&res->bind), usage.resource_stages[index], reg);
		wu32(&w, res->element_size);
		{
			uint8_t shape = resource_shape(&prog->types, res);
			// textures fused with a comparison sampler mark bit 5 themselves
			if (res->kind == svsl_res_texture && res->sampler_slot >= 0)
				for (int32_t k = 0; k < prog->resources.count; k++) {
					const svsl_resource_t *smp = &prog->resources.items[k];
					if (smp->kind == svsl_res_sampler && smp->bind.slot == res->sampler_slot &&
					    smp->bind.space == res->bind.space &&
					    svsl_type_get(&prog->types, smp->type)->is_comparison)
						shape |= 1 << 5;
				}
			const svsl_type_t *rt = svsl_type_get(&prog->types, res->type);
			if (rt->kind == svsl_type_array) rt = svsl_type_get(&prog->types, rt->elem);
			uint8_t format = rt->kind == svsl_type_image
			               ? (uint8_t)svsl_image_format_for(&prog->types, rt) : 0;
			wu8(&w, shape);
			wu8(&w, format);
			wu16(&w, 0); // reserved
		}
	}

	{
		for (int32_t i = 0; i < prog->spec_consts.count; i++) {
			const svsl_spec_const_t *sc = &prog->spec_consts.items[i];
			uint16_t type_count;
			wstr(&w, sc->name, 32);
			wu32(&w, sc->id);
			wu32(&w, sc->default_bits);
			// spec-const bools stay OpTypeBool in SPIR-V; sksc reflects those as
			// int (VkBool32), unlike buffer members which are stored as uint
			uint16_t sc_type = var_type_of(&prog->types, sc->type, &type_count);
			if (svsl_type_get(&prog->types, sc->type)->scalar == svsl_scalar_bool)
				sc_type = var_int;
			wu16(&w, sc_type);
			wu8(&w, usage.spec_stages[i]);
		}
	}

	for (int32_t i = 0; i < module->func_count; i++) {
		wi32(&w, 1); // skr_shader_lang_spirv
		wi32(&w, (int32_t)module->funcs[i].entry->stage);
		wu32(&w, (uint32_t)module->funcs[i].entry->wave_size); // per-entry wave size
		write_stage_spirv(&w, &blobs[i], module->funcs[i].entry->stage);
	}

	out_blob->bytes = w.out.items;
	out_blob->size  = w.out.count;
}
