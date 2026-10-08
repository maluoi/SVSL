// svsl_isa - headless GPU measurement of SPIR-V stages on a real driver: what the
// hardware pays, which SPIR-V word count only loosely proxies (drivers re-optimize).
// Runs anywhere Vulkan does - desktop RADV/ACO and Android Adreno alike.
//
//   svsl_isa [-v] <stage.spv>...
//       The driver's own statistics for each stage (VK_KHR_pipeline_executable_
//       properties): instructions, registers, spills/scratch, code size. -v prints
//       every statistic the driver reports.
//   svsl_isa time [-reps N] <stage.spv>...
//       Cold pipeline-creation time, files interleaved so clock drift spreads across
//       them. Each creation gets a fresh SPIR-V generator word, so no driver cache can
//       hit. Adreno's compile time is shape-sensitive (docs/dev/case-study-astc-encoders.md).
//   svsl_isa run <shader.comp.spv> -groups X Y Z [options]
//       A timed compute dispatch. Resources are reflected from the module and filled
//       deterministically: storage buffers with hashed floats in [0, 1), images with
//       hashed bytes (or -image), uniform buffers with zeros (or -u32/-f32 words). Prints
//       the median dispatch time and a hash of what the dispatch wrote, so two
//       compilers' outputs compare by hash.
//         -iters N        timed dispatches (default 30; the first 4 of >8 warm up)
//         -warm MS        sustained load first, so the GPU clock settles (default 0)
//         -tex W H        image size (default 1024 x 1024)
//         -image FILE     sampled image contents: u32 w, u32 h, u32 fmt (0 rgba8,
//                         1 rgba32f), then the pixels
//         -buf BYTES      storage buffer size (default 16 MiB)
//         -u32 I=V        uniform word I as a uint (repeatable); -f32 I=V as a float
//         -out FILE       writes the first storage buffer after the last dispatch
//
// The pipeline layout is reflected from each module. A fragment stage pairs with its
// sibling .vert.spv (or a trivial embedded vertex shader). Dev-only: links libvulkan,
// unlike the zero-dependency core library. Build with SVSL_BUILD_ISA (CLAUDE.md has
// the Android cross-build).

#define _POSIX_C_SOURCE 199309L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "../vendor/spirv.h"

// A trivial passthrough vertex shader (writes gl_Position), used to form a
// graphics pipeline when the measured stage is a fragment shader.
static const uint32_t MIN_VS[] = {
	0x07230203u,0x00010300u,0x00000000u,0x0000000du,0x00000000u,0x00020011u,
	0x00000001u,0x0006000bu,0x00000001u,0x4c534c47u,0x6474732eu,0x3035342eu,
	0x00000000u,0x0003000eu,0x00000000u,0x00000001u,0x0005000fu,0x00000000u,
	0x00000004u,0x00007376u,0x00000009u,0x00030005u,0x00000004u,0x00007376u,
	0x00030005u,0x00000009u,0x00007376u,0x00040047u,0x00000009u,0x0000000bu,
	0x00000000u,0x00020013u,0x00000002u,0x00030021u,0x00000003u,0x00000002u,
	0x00030016u,0x00000006u,0x00000020u,0x00040017u,0x00000007u,0x00000006u,
	0x00000004u,0x00040020u,0x00000008u,0x00000003u,0x00000007u,0x0004003bu,
	0x00000008u,0x00000009u,0x00000003u,0x0004002bu,0x00000006u,0x0000000au,
	0x00000000u,0x0004002bu,0x00000006u,0x0000000bu,0x3f800000u,0x00050036u,
	0x00000002u,0x00000004u,0x00000000u,0x00000003u,0x000200f8u,0x00000005u,
	0x00070050u,0x00000007u,0x0000000cu,0x0000000au,0x0000000au,0x0000000au,
	0x0000000bu,0x0003003eu,0x00000009u,0x0000000cu,0x000100fdu,0x00010038u,
};

static double now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int cmp_double(const void *a, const void *b) {
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static void *load_file(const char *path, size_t *out_size) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *data = size > 0 ? malloc((size_t)size) : NULL;
	if (!data || fread(data, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(data); return NULL; }
	fclose(f);
	*out_size = (size_t)size;
	return data;
}

static uint32_t *load_spv(const char *path, size_t *out_words) {
	size_t    size  = 0;
	uint32_t *words = load_file(path, &size);
	if (!words || size % 4 != 0 || size < 20 || words[0] != SpvMagicNumber) { free(words); return NULL; }
	*out_words = size / 4;
	return words;
}

// the deterministic fill value for word/byte i (Knuth's multiplicative hash)
static uint32_t hash_u32(uint32_t i) { return i * 2654435761u; }

// --- SPIR-V reflection ------------------------------------------------------
// Just enough to build a compatible pipeline layout and resources: the entry
// point, each descriptor binding (set, binding, type, count, and an image's
// dimension and format), and whether push constants are used.

enum { tk_other, tk_image, tk_sampler, tk_sampled_image, tk_struct, tk_array, tk_runtime_array };

typedef struct type_info_t {
	uint8_t  kind;      // tk_*
	uint8_t  sampled;   // image: 1 = sampled, 2 = storage
	uint32_t dim;       // image: SPIR-V Dim
	uint32_t format;    // image: SPIR-V ImageFormat
	uint32_t elem;      // array/sampled-image/pointer: element, image or pointee type id
	uint32_t len;       // array: length (a resolved constant)
} type_info_t;

typedef struct binding_t {
	uint32_t         set, binding, count;
	VkDescriptorType type;
	uint32_t         dim;    // images: SPIR-V Dim
	uint32_t         format; // images: SPIR-V ImageFormat (0 = unknown)
} binding_t;

typedef struct reflect_t {
	VkShaderStageFlagBits stage;
	char                  entry[64];
	binding_t             bindings[64];
	uint32_t              binding_count;
	uint32_t              set_count;
	bool                  has_push;
} reflect_t;

static VkDescriptorType descriptor_type(const type_info_t *t, uint32_t storage) {
	if (storage == SpvStorageClassUniform)       return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	if (storage == SpvStorageClassStorageBuffer) return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	switch (t->kind) {                                          // UniformConstant: by the pointee
	case tk_sampler:       return VK_DESCRIPTOR_TYPE_SAMPLER;
	case tk_sampled_image: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	case tk_image:         return t->sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	default:               return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	}
}

static bool reflect(const uint32_t *w, size_t n, reflect_t *out) {
	uint32_t     bound     = w[3];
	type_info_t *ti        = calloc(bound, sizeof(type_info_t));
	int32_t     *deco_set  = malloc(bound * sizeof(int32_t));
	int32_t     *deco_bind = malloc(bound * sizeof(int32_t));
	uint32_t    *const_val = calloc(bound, sizeof(uint32_t));
	for (uint32_t i = 0; i < bound; i++) { deco_set[i] = -1; deco_bind[i] = -1; }
	*out = (reflect_t){0};

	// pass 1: types, decorations, constants, the entry point
	for (size_t p = 5; p < n;) {
		uint32_t len = w[p] >> 16, op = w[p] & 0xffff;
		if (len == 0 || p + len > n) break;
		const uint32_t *a = &w[p];
		switch (op) {
		case SpvOpEntryPoint: // model, id, name
			if (!out->stage) {
				out->stage = a[1] == SpvExecutionModelVertex    ? VK_SHADER_STAGE_VERTEX_BIT
				           : a[1] == SpvExecutionModelFragment  ? VK_SHADER_STAGE_FRAGMENT_BIT
				           : a[1] == SpvExecutionModelGLCompute ? VK_SHADER_STAGE_COMPUTE_BIT : 0;
				snprintf(out->entry, sizeof(out->entry), "%.*s", (int)((len - 3) * 4), (const char *)&a[3]);
			}
			break;
		case SpvOpDecorate: // target, decoration, operand
			if (a[1] < bound && a[2] == SpvDecorationDescriptorSet) deco_set[a[1]]  = (int32_t)a[3];
			if (a[1] < bound && a[2] == SpvDecorationBinding)       deco_bind[a[1]] = (int32_t)a[3];
			break;
		case SpvOpConstant:         if (a[2] < bound) const_val[a[2]] = a[3]; break;
		case SpvOpTypeImage:        if (a[1] < bound) ti[a[1]] = (type_info_t){ .kind = tk_image, .dim = a[3], .sampled = (uint8_t)a[7], .format = a[8] }; break;
		case SpvOpTypeSampler:      if (a[1] < bound) ti[a[1]].kind = tk_sampler; break;
		case SpvOpTypeSampledImage: if (a[1] < bound) ti[a[1]] = (type_info_t){ .kind = tk_sampled_image, .elem = a[2] }; break;
		case SpvOpTypeArray:        if (a[1] < bound) ti[a[1]] = (type_info_t){ .kind = tk_array, .elem = a[2], .len = a[3] < bound ? const_val[a[3]] : 1 }; break;
		case SpvOpTypeRuntimeArray: if (a[1] < bound) ti[a[1]] = (type_info_t){ .kind = tk_runtime_array, .elem = a[2] }; break;
		case SpvOpTypeStruct:       if (a[1] < bound) ti[a[1]].kind = tk_struct; break;
		case SpvOpTypePointer:      if (a[1] < bound) ti[a[1]].elem = a[3]; break; // pointee
		default: break;
		}
		p += len;
	}

	// pass 2: descriptor variables
	for (size_t p = 5; p < n;) {
		uint32_t len = w[p] >> 16, op = w[p] & 0xffff;
		if (len == 0 || p + len > n) break;
		const uint32_t *a = &w[p];
		p += len;
		if (op != SpvOpVariable) continue; // pointer type, id, storage class
		uint32_t ptr = a[1], var = a[2], storage = a[3];
		if (storage == SpvStorageClassPushConstant) out->has_push = true;
		if ((storage != SpvStorageClassUniformConstant && storage != SpvStorageClassUniform &&
		     storage != SpvStorageClassStorageBuffer) || var >= bound || deco_set[var] < 0 || deco_bind[var] < 0)
			continue;
		uint32_t pointee = ptr < bound ? ti[ptr].elem : 0, count = 1;
		while (pointee < bound && (ti[pointee].kind == tk_array || ti[pointee].kind == tk_runtime_array)) {
			if (ti[pointee].kind == tk_array && ti[pointee].len) count = ti[pointee].len;
			pointee = ti[pointee].elem;
		}
		if (pointee >= bound || out->binding_count >= 64) continue;
		const type_info_t *t   = &ti[pointee];
		const type_info_t *img = t->kind == tk_sampled_image && t->elem < bound ? &ti[t->elem] : t;
		out->bindings[out->binding_count++] = (binding_t){
			.set = (uint32_t)deco_set[var], .binding = (uint32_t)deco_bind[var], .count = count,
			.type = descriptor_type(t, storage), .dim = img->dim, .format = img->format };
		if ((uint32_t)deco_set[var] + 1 > out->set_count) out->set_count = (uint32_t)deco_set[var] + 1;
	}

	free(ti); free(deco_set); free(deco_bind); free(const_val);
	return out->stage != 0 && out->set_count <= 8;
}

// --- device -----------------------------------------------------------------

static VkInstance       g_inst;
static VkPhysicalDevice g_phys;
static VkDevice         g_dev;
static VkQueue          g_queue;
static uint32_t         g_qfam;
static float            g_ts_period;
static bool             g_has_stats;
static PFN_vkGetPipelineExecutablePropertiesKHR g_props;
static PFN_vkGetPipelineExecutableStatisticsKHR g_stats;

static bool has_extension(const VkExtensionProperties *exts, uint32_t count, const char *name) {
	for (uint32_t i = 0; i < count; i++)
		if (strcmp(exts[i].extensionName, name) == 0) return true;
	return false;
}

// Creates the device with every feature it supports (shaders may use 16-bit
// storage, float16, int64...) except robustness, which changes the generated code.
static bool vk_init(void) {
	VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
	if (vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	                                              .pApplicationInfo = &app }, NULL, &g_inst) != VK_SUCCESS) {
		fprintf(stderr, "vkCreateInstance failed\n");
		return false;
	}
	uint32_t         count = 16;
	VkPhysicalDevice devs[16];
	vkEnumeratePhysicalDevices(g_inst, &count, devs);
	for (uint32_t i = 0; i < count && !g_phys; i++) {
		VkPhysicalDeviceProperties pr;
		vkGetPhysicalDeviceProperties(devs[i], &pr);
		if (pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) g_phys = devs[i];
	}
	if (!g_phys) { fprintf(stderr, "no hardware GPU found\n"); return false; }
	VkPhysicalDeviceProperties pr;
	vkGetPhysicalDeviceProperties(g_phys, &pr);
	g_ts_period = pr.limits.timestampPeriod;
	fprintf(stderr, "device: %s (driver 0x%x, Vulkan %u.%u)\n", pr.deviceName, pr.driverVersion,
	        VK_API_VERSION_MAJOR(pr.apiVersion), VK_API_VERSION_MINOR(pr.apiVersion));

	uint32_t                qcount = 16;
	VkQueueFamilyProperties qprops[16];
	vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qcount, qprops);
	for (uint32_t i = qcount; i-- > 0;)
		if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) g_qfam = i;

	uint32_t ext_count = 0;
	vkEnumerateDeviceExtensionProperties(g_phys, NULL, &ext_count, NULL);
	VkExtensionProperties *exts = malloc(ext_count * sizeof(*exts));
	vkEnumerateDeviceExtensionProperties(g_phys, NULL, &ext_count, exts);
	const char *enable[2];
	uint32_t    enable_count = 0;
	g_has_stats = has_extension(exts, ext_count, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
	if (g_has_stats) enable[enable_count++] = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
	bool v13 = pr.apiVersion >= VK_API_VERSION_1_3;
	if (!v13 && has_extension(exts, ext_count, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
		enable[enable_count++] = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
	free(exts);

	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
		.pipelineExecutableInfo = VK_TRUE };
	VkPhysicalDeviceDynamicRenderingFeatures dr = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, .dynamicRendering = VK_TRUE,
		.pNext = g_has_stats ? &pe : NULL };
	VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
	VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = v13 ? (void *)&f13 : NULL };
	VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12 };
	VkPhysicalDeviceFeatures2        f   = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11 };
	vkGetPhysicalDeviceFeatures2(g_phys, &f);
	f.features.robustBufferAccess = VK_FALSE;
	f11.protectedMemory           = VK_FALSE;
	f13.robustImageAccess         = VK_FALSE;
	f13.pNext = g_has_stats ? &pe : NULL;
	if (!v13) f12.pNext = &dr;

	float prio = 1.0f;
	VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f,
		.queueCreateInfoCount = 1, .enabledExtensionCount = enable_count, .ppEnabledExtensionNames = enable,
		.pQueueCreateInfos = &(VkDeviceQueueCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = g_qfam, .queueCount = 1, .pQueuePriorities = &prio } };
	if (vkCreateDevice(g_phys, &dci, NULL, &g_dev) != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed\n"); return false; }
	vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);
	g_props = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutablePropertiesKHR");
	g_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutableStatisticsKHR");
	g_has_stats = g_has_stats && g_props && g_stats;
	return true;
}

// --- pipelines --------------------------------------------------------------

typedef struct shader_t {
	const char           *file;
	uint32_t             *spv;
	size_t                words;
	reflect_t             r;
	VkDescriptorSetLayout sets[8];
	VkPipelineLayout      layout;
} shader_t;

static bool shader_load(const char *file, shader_t *out) {
	*out = (shader_t){ .file = file };
	out->spv = load_spv(file, &out->words);
	if (!out->spv)                               { fprintf(stderr, "cannot read SPIR-V from %s\n", file); return false; }
	if (!reflect(out->spv, out->words, &out->r)) { fprintf(stderr, "cannot reflect %s\n", file); return false; }
	for (uint32_t s = 0; s < out->r.set_count; s++) {
		VkDescriptorSetLayoutBinding b[64];
		uint32_t                     bc = 0;
		for (uint32_t i = 0; i < out->r.binding_count; i++)
			if (out->r.bindings[i].set == s)
				b[bc++] = (VkDescriptorSetLayoutBinding){ .binding = out->r.bindings[i].binding,
					.descriptorType = out->r.bindings[i].type, .descriptorCount = out->r.bindings[i].count,
					.stageFlags = VK_SHADER_STAGE_ALL };
		vkCreateDescriptorSetLayout(g_dev, &(VkDescriptorSetLayoutCreateInfo){
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = bc, .pBindings = b },
			NULL, &out->sets[s]);
	}
	VkPushConstantRange pc = { .stageFlags = VK_SHADER_STAGE_ALL, .size = 128 };
	vkCreatePipelineLayout(g_dev, &(VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = out->r.set_count, .pSetLayouts = out->sets,
		.pushConstantRangeCount = out->r.has_push ? 1 : 0, .pPushConstantRanges = &pc }, NULL, &out->layout);
	return true;
}

static void shader_free(shader_t *s) {
	if (s->layout) vkDestroyPipelineLayout(g_dev, s->layout, NULL);
	for (uint32_t i = 0; i < s->r.set_count; i++) vkDestroyDescriptorSetLayout(g_dev, s->sets[i], NULL);
	free(s->spv);
}

static VkShaderModule make_module(const uint32_t *w, size_t words) {
	VkShaderModule m = VK_NULL_HANDLE;
	vkCreateShaderModule(g_dev, &(VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = words * 4, .pCode = w }, NULL, &m);
	return m;
}

// The stage's pipeline; *out_ms is the vkCreate*Pipelines time alone. A fragment
// stage pairs with its sibling .vert.spv, so interpolated inputs are defined (a
// driver would treat them as undefined and remove the dependent work).
static VkPipeline create_pipeline(const shader_t *s, bool capture, double *out_ms) {
	VkPipelineCreateFlags flags = capture && g_has_stats ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
	VkShaderModule        mod   = make_module(s->spv, s->words);
	VkPipeline            pipe  = VK_NULL_HANDLE;
	double                t0    = 0;

	if (s->r.stage == VK_SHADER_STAGE_COMPUTE_BIT) {
		VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.flags = flags, .layout = s->layout,
			.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			           .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = s->r.entry } };
		t0 = now_ms();
		vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
	} else {
		bool           frag     = s->r.stage == VK_SHADER_STAGE_FRAGMENT_BIT;
		VkShaderModule vs_mod   = mod;
		const char    *vs_entry = s->r.entry;
		uint32_t      *sib      = NULL;
		reflect_t      vr       = {0};
		if (frag) {
			char   vspath[1024];
			size_t sw  = 0;
			snprintf(vspath, sizeof(vspath), "%s", s->file);
			char  *dot = strstr(vspath, ".frag.spv");
			if (dot) { memcpy(dot, ".vert.spv", 9); sib = load_spv(vspath, &sw); }
			if (sib && reflect(sib, sw, &vr)) { vs_mod = make_module(sib, sw); vs_entry = vr.entry; }
			else                              { vs_mod = make_module(MIN_VS, sizeof(MIN_VS) / 4); vs_entry = "vs"; }
		}
		VkPipelineShaderStageCreateInfo stages[2] = {
			{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
			  .module = vs_mod, .pName = vs_entry },
			{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			  .module = mod, .pName = s->r.entry } };
		VkFormat                            color = VK_FORMAT_R8G8B8A8_UNORM;
		VkPipelineColorBlendAttachmentState cba   = { .colorWriteMask = 0xf };
		VkDynamicState                      dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkGraphicsPipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &(VkPipelineRenderingCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
				.colorAttachmentCount = frag ? 1 : 0, .pColorAttachmentFormats = &color },
			.flags = flags, .stageCount = frag ? 2 : 1, .pStages = stages, .layout = s->layout,
			.pVertexInputState   = &(VkPipelineVertexInputStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO },
			.pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST },
			.pViewportState      = &(VkPipelineViewportStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1, .scissorCount = 1 },
			.pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.rasterizerDiscardEnable = frag ? VK_FALSE : VK_TRUE, .polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f },
			.pMultisampleState   = &(VkPipelineMultisampleStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT },
			.pColorBlendState    = &(VkPipelineColorBlendStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = frag ? 1 : 0, .pAttachments = &cba },
			.pDynamicState       = &(VkPipelineDynamicStateCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = 2, .pDynamicStates = dyn } };
		t0 = now_ms();
		vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
		if (frag) { vkDestroyShaderModule(g_dev, vs_mod, NULL); free(sib); }
	}
	if (out_ms) *out_ms = now_ms() - t0;
	vkDestroyShaderModule(g_dev, mod, NULL);
	return pipe;
}

// --- statistics -------------------------------------------------------------

// Driver statistic names (RADV/ACO and Adreno spell them differently) -> the short
// keys of the summary line, in print order. A name matches by prefix; the first
// matching row wins, so the spill rows sit before the plain register rows.
static const struct { const char *name, *key; } stat_keys[] = {
	{ "Instruction count all",                          "insts"   }, // Adreno
	{ "Instructions",                                   "insts"   }, // RADV
	{ "ALU instruction count 32bit",                    "alu32"   },
	{ "Overall register footprint",                     "regs"    },
	{ "Spilled VGPRs",                                  "spill_v" },
	{ "Spilled SGPRs",                                  "spill_s" },
	{ "VGPRs",                                          "vgpr"    },
	{ "SGPRs",                                          "sgpr"    },
	{ "Scratch memory usage",                           "scratch" },
	{ "Scratch size",                                   "scratch" },
	{ "Memory read instruction count",                  "mem_rd"  },
	{ "Memory write instruction count",                 "mem_wr"  },
	{ "Code size",                                      "code"    },
};
#define STAT_KEY_COUNT (sizeof(stat_keys) / sizeof(stat_keys[0]))

static double stat_value(const VkPipelineExecutableStatisticKHR *st) {
	switch (st->format) {
	case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:  return st->value.b32;
	case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:   return (double)st->value.i64;
	case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:  return (double)st->value.u64;
	case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: return st->value.f64;
	default:                                                  return 0;
	}
}

// One summary line for the executable that runs `stage`, plus every statistic with -v.
static void print_stats(const char *file, VkPipeline pipe, VkShaderStageFlagBits stage, bool all) {
	const char *sname = stage == VK_SHADER_STAGE_VERTEX_BIT ? "vertex" : stage == VK_SHADER_STAGE_FRAGMENT_BIT ? "fragment" : "compute";
	if (!g_has_stats) { printf("ISA %s stage=%s (driver has no pipeline executable statistics)\n", file, sname); return; }
	VkPipelineInfoKHR pinfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe };
	uint32_t          nexec = 0;
	g_props(g_dev, &pinfo, &nexec, NULL);
	VkPipelineExecutablePropertiesKHR *props = calloc(nexec ? nexec : 1, sizeof(*props));
	for (uint32_t i = 0; i < nexec; i++) props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
	g_props(g_dev, &pinfo, &nexec, props);

	uint32_t e = 0;
	while (e < nexec && !(props[e].stages & stage)) e++;
	if (e == nexec) { printf("ISA %s stage=%s (no executable for the stage)\n", file, sname); free(props); return; }
	VkPipelineExecutableInfoKHR ei = { .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipe, .executableIndex = e };
	uint32_t ns = 0;
	g_stats(g_dev, &ei, &ns, NULL);
	VkPipelineExecutableStatisticKHR *st = calloc(ns ? ns : 1, sizeof(*st));
	for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
	g_stats(g_dev, &ei, &ns, st);

	double value[STAT_KEY_COUNT];
	bool   found[STAT_KEY_COUNT] = {0};
	for (uint32_t i = 0; i < ns; i++)
		for (uint32_t k = 0; k < STAT_KEY_COUNT; k++)
			if (strncmp(st[i].name, stat_keys[k].name, strlen(stat_keys[k].name)) == 0) {
				if (!found[k]) { found[k] = true; value[k] = stat_value(&st[i]); }
				break;
			}
	printf("ISA %s stage=%s", file, sname);
	for (uint32_t k = 0; k < STAT_KEY_COUNT; k++)
		if (found[k]) printf(" %s=%.0f", stat_keys[k].key, value[k]);
	printf("\n");
	for (uint32_t i = 0; all && i < ns; i++) {
		if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR) printf("  %s = %g\n", st[i].name, st[i].value.f64);
		else printf("  %s = %llu\n", st[i].name, (unsigned long long)(st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR ? st[i].value.b32 : st[i].value.u64));
	}
	free(st);
	free(props);
}

static int cmd_stats(int argc, char **argv) {
	bool all = false;
	int  rc  = 0;
	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) { all = true; continue; }
		shader_t s;
		if (!shader_load(argv[i], &s)) { shader_free(&s); rc = 1; continue; }
		VkPipeline pipe = create_pipeline(&s, true, NULL);
		if (pipe) print_stats(s.file, pipe, s.r.stage, all);
		else      { printf("ISA %s (pipeline creation failed)\n", s.file); rc = 1; }
		vkDestroyPipeline(g_dev, pipe, NULL);
		shader_free(&s);
	}
	return rc;
}

// --- time: cold pipeline creation -------------------------------------------

static int cmd_time(int argc, char **argv) {
	int32_t reps = 3, n = 0;
	shader_t *shaders = calloc((size_t)argc, sizeof(shader_t));
	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-reps") == 0 && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
		if (!shader_load(argv[i], &shaders[n])) { shader_free(&shaders[n]); return 1; }
		n++;
	}
	if (reps < 1) reps = 1;
	double  *ms   = calloc((size_t)n * (size_t)reps, sizeof(double));
	uint32_t seed = (uint32_t)(now_ms() * 1000.0);
	for (int32_t r = 0; r < reps; r++)
		for (int32_t i = 0; i < n; i++) {
			shaders[i].spv[2] = seed + (uint32_t)(r * n + i); // the generator word: a new hash every time
			VkPipeline pipe = create_pipeline(&shaders[i], false, &ms[i * reps + r]);
			if (!pipe) { printf("TIME %s (pipeline creation failed)\n", shaders[i].file); return 1; }
			vkDestroyPipeline(g_dev, pipe, NULL);
		}
	for (int32_t i = 0; i < n; i++) {
		double *m = &ms[i * reps];
		qsort(m, (size_t)reps, sizeof(double), cmp_double);
		printf("TIME %s pipeline median %.2f ms min %.2f max %.2f (n=%d)\n", shaders[i].file, m[reps / 2], m[0], m[reps - 1], reps);
		shader_free(&shaders[i]);
	}
	free(ms);
	free(shaders);
	return 0;
}

// --- run: a timed compute dispatch ------------------------------------------

typedef struct buf_t { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize size; } buf_t;
typedef struct img_t { VkImage img; VkDeviceMemory mem; VkImageView view; VkFormat format; uint32_t w, h, bpp; } img_t;

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want) {
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
	return 0;
}

static buf_t make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
	buf_t b = { .size = size };
	vkCreateBuffer(g_dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage }, NULL, &b.buf);
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(g_dev, b.buf, &req);
	VkMemoryPropertyFlags want = host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	                                  : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	vkAllocateMemory(g_dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = req.size, .memoryTypeIndex = find_mem(req.memoryTypeBits, want) }, NULL, &b.mem);
	vkBindBufferMemory(g_dev, b.buf, b.mem, 0);
	if (host) vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map);
	return b;
}

static void free_buffer(buf_t *b) {
	vkDestroyBuffer(g_dev, b->buf, NULL);
	vkFreeMemory(g_dev, b->mem, NULL);
}

// SPIR-V ImageFormat -> the Vulkan format a storage image gets (bytes per texel); unknown -> rgba8
static VkFormat storage_format(uint32_t spv_format, uint32_t *out_bpp) {
	static const struct { uint32_t spv; VkFormat vk; uint32_t bpp; } formats[] = {
		{ SpvImageFormatRgba32f,    VK_FORMAT_R32G32B32A32_SFLOAT, 16 },
		{ SpvImageFormatRgba16f,    VK_FORMAT_R16G16B16A16_SFLOAT,  8 },
		{ SpvImageFormatR32f,       VK_FORMAT_R32_SFLOAT,           4 },
		{ SpvImageFormatRgba8,      VK_FORMAT_R8G8B8A8_UNORM,       4 },
		{ SpvImageFormatRgba8Snorm, VK_FORMAT_R8G8B8A8_SNORM,       4 },
		{ SpvImageFormatRg32f,      VK_FORMAT_R32G32_SFLOAT,        8 },
		{ SpvImageFormatRg16f,      VK_FORMAT_R16G16_SFLOAT,        4 },
		{ SpvImageFormatR16f,       VK_FORMAT_R16_SFLOAT,           2 },
		{ SpvImageFormatR8,         VK_FORMAT_R8_UNORM,             1 },
		{ SpvImageFormatRgba32i,    VK_FORMAT_R32G32B32A32_SINT,   16 },
		{ SpvImageFormatR32i,       VK_FORMAT_R32_SINT,             4 },
		{ SpvImageFormatRgba32ui,   VK_FORMAT_R32G32B32A32_UINT,   16 },
		{ SpvImageFormatR32ui,      VK_FORMAT_R32_UINT,             4 },
		{ SpvImageFormatRg32ui,     VK_FORMAT_R32G32_UINT,          8 },
	};
	for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
		if (formats[i].spv == spv_format) { *out_bpp = formats[i].bpp; return formats[i].vk; }
	*out_bpp = 4;
	return VK_FORMAT_R8G8B8A8_UNORM;
}

static img_t make_image(uint32_t w, uint32_t h, VkFormat format, uint32_t bpp, VkImageUsageFlags usage) {
	img_t im = { .format = format, .w = w, .h = h, .bpp = bpp };
	vkCreateImage(g_dev, &(VkImageCreateInfo){ .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
		.format = format, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT },
		NULL, &im.img);
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(g_dev, im.img, &req);
	vkAllocateMemory(g_dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = req.size, .memoryTypeIndex = find_mem(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) },
		NULL, &im.mem);
	vkBindImageMemory(g_dev, im.img, im.mem, 0);
	vkCreateImageView(g_dev, &(VkImageViewCreateInfo){ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.img,
		.viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
		NULL, &im.view);
	return im;
}

static void image_barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to) {
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1,
		&(VkImageMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			.oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } });
}

static void memory_barrier(VkCommandBuffer cmd) {
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
		&(VkMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
		                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT }, 0, NULL, 0, NULL);
}

// one-shot command buffer helper: begin, then submit-and-wait
typedef struct cmd_t { VkCommandPool pool; VkCommandBuffer cb; VkFence fence; } cmd_t;

static void cmd_begin(cmd_t *c) {
	vkResetFences(g_dev, 1, &c->fence);
	vkResetCommandBuffer(c->cb, 0);
	vkBeginCommandBuffer(c->cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO });
}
static void cmd_submit(cmd_t *c) {
	vkEndCommandBuffer(c->cb);
	vkQueueSubmit(g_queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
	                                           .pCommandBuffers = &c->cb }, c->fence);
	vkWaitForFences(g_dev, 1, &c->fence, VK_TRUE, UINT64_MAX);
}

static uint64_t fnv1a(uint64_t h, const uint8_t *data, size_t size) {
	for (size_t i = 0; i < size; i++) h = (h ^ data[i]) * 0x100000001b3ull;
	return h;
}

// the Adreno GPU clock (kgsl), or 0 where the driver doesn't expose it
static double gpu_clock_mhz(void) {
	FILE     *f  = fopen("/sys/class/kgsl/kgsl-3d0/gpuclk", "r");
	long long hz = 0;
	if (!f) return 0;
	if (fscanf(f, "%lld", &hz) != 1) hz = 0;
	fclose(f);
	return (double)hz / 1e6;
}

typedef struct run_opts_t {
	uint32_t     groups[3];
	int32_t      iters, warm_ms;
	uint32_t     tex_w, tex_h;
	const char  *image, *out;
	VkDeviceSize buf_size;
	uint32_t     ubo_words[256]; // -u32/-f32 overrides
} run_opts_t;

static int cmd_run(int argc, char **argv) {
	run_opts_t o = { .iters = 30, .tex_w = 1024, .tex_h = 1024, .buf_size = 16u << 20 };
	const char *file = NULL;
	for (int i = 0; i < argc; i++) {
		const char *a = argv[i];
		bool more1 = i + 1 < argc, more3 = i + 3 < argc;
		if      (strcmp(a, "-groups") == 0 && more3) { for (int k = 0; k < 3; k++) o.groups[k] = (uint32_t)atoi(argv[++i]); }
		else if (strcmp(a, "-iters")  == 0 && more1) o.iters    = atoi(argv[++i]);
		else if (strcmp(a, "-warm")   == 0 && more1) o.warm_ms  = atoi(argv[++i]);
		else if (strcmp(a, "-tex")    == 0 && i + 2 < argc) { o.tex_w = (uint32_t)atoi(argv[++i]); o.tex_h = (uint32_t)atoi(argv[++i]); }
		else if (strcmp(a, "-image")  == 0 && more1) o.image    = argv[++i];
		else if (strcmp(a, "-buf")    == 0 && more1) o.buf_size = (VkDeviceSize)strtoull(argv[++i], NULL, 10);
		else if (strcmp(a, "-out")    == 0 && more1) o.out      = argv[++i];
		else if ((strcmp(a, "-u32") == 0 || strcmp(a, "-f32") == 0) && more1) {
			const char *arg = argv[++i], *eq = strchr(arg, '=');
			long        idx = atol(arg);
			if (!eq || idx < 0 || idx >= 256) { fprintf(stderr, "bad %s %s (want I=V, I < 256)\n", a, arg); return 2; }
			if (a[1] == 'u') o.ubo_words[idx] = (uint32_t)strtoul(eq + 1, NULL, 0);
			else { float f = strtof(eq + 1, NULL); memcpy(&o.ubo_words[idx], &f, 4); }
		}
		else if (a[0] != '-' && !file) file = a;
		else { fprintf(stderr, "run: unknown or incomplete option '%s'\n", a); return 2; }
	}
	if (!file || !o.groups[0] || !o.groups[1] || !o.groups[2]) { fprintf(stderr, "run: need <shader.comp.spv> -groups X Y Z\n"); return 2; }
	if (o.iters < 1) o.iters = 1;

	shader_t s;
	if (!shader_load(file, &s)) return 1;
	if (s.r.stage != VK_SHADER_STAGE_COMPUTE_BIT) { fprintf(stderr, "run: %s is not a compute shader\n", file); return 1; }
	VkPipeline pipe = create_pipeline(&s, false, NULL);
	if (!pipe) { fprintf(stderr, "run: pipeline creation failed\n"); return 1; }

	cmd_t c = {0};
	vkCreateCommandPool(g_dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.queueFamilyIndex = g_qfam, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &c.pool);
	vkAllocateCommandBuffers(g_dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = c.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &c.cb);
	vkCreateFence(g_dev, &(VkFenceCreateInfo){ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }, NULL, &c.fence);

	// the sampled image: -image contents, or hashed rgba8 bytes
	uint32_t  sw = o.tex_w, sh = o.tex_h, sbpp = 4;
	uint8_t  *spixels = NULL;
	VkFormat  sformat = VK_FORMAT_R8G8B8A8_UNORM;
	if (o.image) {
		size_t   size = 0;
		uint8_t *raw  = load_file(o.image, &size);
		if (!raw || size < 12) { fprintf(stderr, "run: cannot read %s\n", o.image); return 1; }
		uint32_t hdr[3];
		memcpy(hdr, raw, 12);
		sw = hdr[0]; sh = hdr[1]; sbpp = hdr[2] ? 16 : 4;
		sformat = hdr[2] ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
		if (size < 12 + (size_t)sw * sh * sbpp) { fprintf(stderr, "run: %s is truncated\n", o.image); return 1; }
		spixels = malloc((size_t)sw * sh * sbpp);
		memcpy(spixels, raw + 12, (size_t)sw * sh * sbpp);
		free(raw);
	}

	// resources per binding (an arrayed binding repeats its one resource)
	buf_t     bufs[64] = {0};
	img_t     imgs[64] = {0};
	VkSampler sampler  = VK_NULL_HANDLE;
	vkCreateSampler(g_dev, &(VkSamplerCreateInfo){ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.maxLod = VK_LOD_CLAMP_NONE }, NULL, &sampler);
	VkDeviceSize staging_size = (VkDeviceSize)sw * sh * sbpp;
	for (uint32_t i = 0; i < s.r.binding_count; i++) {
		const binding_t *b = &s.r.bindings[i];
		if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && o.buf_size > staging_size) staging_size = o.buf_size;
		if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
			uint32_t bpp;
			storage_format(b->format, &bpp);
			if ((VkDeviceSize)o.tex_w * o.tex_h * bpp > staging_size) staging_size = (VkDeviceSize)o.tex_w * o.tex_h * bpp;
		}
		if ((b->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
		     b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) && b->dim != SpvDim2D) {
			fprintf(stderr, "run: binding %u.%u is not a 2D image (dim %u); only 2D images are supported\n", b->set, b->binding, b->dim);
			return 1;
		}
	}
	buf_t staging = make_buffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

	for (uint32_t i = 0; i < s.r.binding_count; i++) {
		const binding_t *b = &s.r.bindings[i];
		switch (b->type) {
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: // 16 KiB of zeros: the smallest maxUniformBufferRange Vulkan allows
			bufs[i] = make_buffer(16384, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
			memset(bufs[i].map, 0, 16384);
			memcpy(bufs[i].map, o.ubo_words, sizeof(o.ubo_words));
			break;
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
			bufs[i] = make_buffer(o.buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
			uint32_t *words = staging.map;
			for (VkDeviceSize k = 0; k < o.buf_size / 4; k++) { // floats in [0, 1): word 0 is 0.0
				float f = (float)(hash_u32((uint32_t)k) & 0xFFFF) / 65536.0f;
				memcpy(&words[k], &f, 4);
			}
			cmd_begin(&c);
			vkCmdCopyBuffer(c.cb, staging.buf, bufs[i].buf, 1, &(VkBufferCopy){ .size = o.buf_size });
			cmd_submit(&c);
			break;
		}
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
			bool     storage = b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			uint32_t bpp     = sbpp;
			VkFormat format  = storage ? storage_format(b->format, &bpp) : sformat;
			uint32_t w = storage ? o.tex_w : sw, h = storage ? o.tex_h : sh;
			imgs[i] = make_image(w, h, format, bpp, storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT);
			if (!storage && spixels) memcpy(staging.map, spixels, (size_t)w * h * bpp);
			else for (size_t k = 0; k < (size_t)w * h * bpp; k++) ((uint8_t *)staging.map)[k] = (uint8_t)(hash_u32((uint32_t)k) >> 16);
			cmd_begin(&c);
			image_barrier(c.cb, imgs[i].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			vkCmdCopyBufferToImage(c.cb, staging.buf, imgs[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
				&(VkBufferImageCopy){ .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { w, h, 1 } });
			image_barrier(c.cb, imgs[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			              storage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			cmd_submit(&c);
			break;
		}
		default:
			break; // samplers share the one sampler
		}
	}

	// descriptor sets
	VkDescriptorPoolSize sizes[64];
	uint32_t             size_count = 0;
	for (uint32_t i = 0; i < s.r.binding_count; i++)
		sizes[size_count++] = (VkDescriptorPoolSize){ .type = s.r.bindings[i].type, .descriptorCount = s.r.bindings[i].count };
	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkDescriptorSet  sets[8];
	vkCreateDescriptorPool(g_dev, &(VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = s.r.set_count ? s.r.set_count : 1, .poolSizeCount = size_count, .pPoolSizes = sizes }, NULL, &pool);
	if (s.r.set_count)
		vkAllocateDescriptorSets(g_dev, &(VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = pool, .descriptorSetCount = s.r.set_count, .pSetLayouts = s.sets }, sets);
	for (uint32_t i = 0; i < s.r.binding_count; i++) {
		const binding_t *b = &s.r.bindings[i];
		VkDescriptorBufferInfo binfo[64];
		VkDescriptorImageInfo  iinfo[64];
		uint32_t               count = b->count < 64 ? b->count : 64;
		for (uint32_t k = 0; k < count; k++) {
			binfo[k] = (VkDescriptorBufferInfo){ .buffer = bufs[i].buf, .range = VK_WHOLE_SIZE };
			iinfo[k] = (VkDescriptorImageInfo){ .sampler = sampler, .imageView = imgs[i].view,
				.imageLayout = b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		}
		bool is_buffer = b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		vkUpdateDescriptorSets(g_dev, 1, &(VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = sets[b->set], .dstBinding = b->binding, .descriptorCount = count, .descriptorType = b->type,
			.pBufferInfo = is_buffer ? binfo : NULL, .pImageInfo = is_buffer ? NULL : iinfo }, 0, NULL);
	}

	// optional sustained load first, so the GPU clock settles where loaded work runs
	if (o.warm_ms > 0) {
		cmd_begin(&c);
		vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
		if (s.r.set_count) vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, s.r.set_count, sets, 0, NULL);
		for (int k = 0; k < 4; k++) vkCmdDispatch(c.cb, o.groups[0], o.groups[1], o.groups[2]);
		vkEndCommandBuffer(c.cb);
		for (double start = now_ms(); now_ms() - start < o.warm_ms;) {
			vkResetFences(g_dev, 1, &c.fence);
			vkQueueSubmit(g_queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
			                                           .pCommandBuffers = &c.cb }, c.fence);
			vkWaitForFences(g_dev, 1, &c.fence, VK_TRUE, UINT64_MAX);
		}
	}

	// one submit per dispatch, so each timestamp pair brackets exactly one dispatch
	VkQueryPool qp;
	vkCreateQueryPool(g_dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 }, NULL, &qp);
	double *ms  = calloc((size_t)o.iters, sizeof(double));
	double  clk = 0;
	for (int32_t it = 0; it < o.iters; it++) {
		cmd_begin(&c);
		vkCmdResetQueryPool(c.cb, qp, 0, 2);
		vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
		if (s.r.set_count) vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, s.r.set_count, sets, 0, NULL);
		vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
		vkCmdDispatch(c.cb, o.groups[0], o.groups[1], o.groups[2]);
		vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
		cmd_submit(&c);
		uint64_t ts[2] = {0};
		vkGetQueryPoolResults(g_dev, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
		ms[it] = (double)(ts[1] - ts[0]) * g_ts_period / 1e6;
		clk   += gpu_clock_mhz();
	}
	int32_t warm = o.iters > 8 ? 4 : 0, n = o.iters - warm;
	qsort(ms + warm, (size_t)n, sizeof(double), cmp_double);

	// what the dispatch wrote: every storage buffer and image, hashed
	uint64_t hash = 0xcbf29ce484222325ull;
	bool     out_written = false;
	for (uint32_t i = 0; i < s.r.binding_count; i++) {
		const binding_t *b = &s.r.bindings[i];
		if (b->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && b->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) continue;
		VkDeviceSize size;
		cmd_begin(&c);
		memory_barrier(c.cb);
		if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
			size = o.buf_size;
			vkCmdCopyBuffer(c.cb, bufs[i].buf, staging.buf, 1, &(VkBufferCopy){ .size = size });
		} else {
			size = (VkDeviceSize)imgs[i].w * imgs[i].h * imgs[i].bpp;
			vkCmdCopyImageToBuffer(c.cb, imgs[i].img, VK_IMAGE_LAYOUT_GENERAL, staging.buf, 1,
				&(VkBufferImageCopy){ .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { imgs[i].w, imgs[i].h, 1 } });
		}
		cmd_submit(&c);
		hash = fnv1a(hash, staging.map, (size_t)size);
		if (o.out && !out_written && b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
			FILE *f = fopen(o.out, "wb");
			if (f) { fwrite(staging.map, 1, (size_t)size, f); fclose(f); }
			out_written = true;
		}
	}

	printf("RUN %s groups %u %u %u median %.3f ms min %.3f p90 %.3f (n=%d)",
	       file, o.groups[0], o.groups[1], o.groups[2], ms[warm + n / 2], ms[warm], ms[warm + (n * 9) / 10], n);
	if (clk > 0) printf(" gpuclk %.0f MHz", clk / o.iters);
	printf(" hash %016llx\n", (unsigned long long)hash);

	free(ms);
	free(spixels);
	vkDestroyQueryPool(g_dev, qp, NULL);
	vkDestroyDescriptorPool(g_dev, pool, NULL);
	for (uint32_t i = 0; i < s.r.binding_count; i++) {
		if (bufs[i].buf) free_buffer(&bufs[i]);
		if (imgs[i].img) { vkDestroyImageView(g_dev, imgs[i].view, NULL); vkDestroyImage(g_dev, imgs[i].img, NULL); vkFreeMemory(g_dev, imgs[i].mem, NULL); }
	}
	free_buffer(&staging);
	vkDestroySampler(g_dev, sampler, NULL);
	vkDestroyFence(g_dev, c.fence, NULL);
	vkDestroyCommandPool(g_dev, c.pool, NULL);
	vkDestroyPipeline(g_dev, pipe, NULL);
	shader_free(&s);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: svsl_isa [-v] <stage.spv>...                 driver statistics\n"
		                "       svsl_isa time [-reps N] <stage.spv>...       cold pipeline-creation time\n"
		                "       svsl_isa run <shader.comp.spv> -groups X Y Z  timed dispatch (see app/isa.c)\n");
		return 2;
	}
	if (!vk_init()) return 1;
	int rc = strcmp(argv[1], "time") == 0 ? cmd_time(argc - 2, argv + 2)
	       : strcmp(argv[1], "run")  == 0 ? cmd_run (argc - 2, argv + 2)
	       :                                cmd_stats(argc - 1, argv + 1);
	vkDestroyDevice(g_dev, NULL);
	vkDestroyInstance(g_inst, NULL);
	return rc;
}
