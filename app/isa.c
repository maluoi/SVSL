// svsl_isa — headless RDNA ISA measurement. Compiles a .spv stage into a real
// RADV pipeline and reports the hardware cost the GPU actually pays: VGPR/SGPR
// register usage (→ occupancy), instruction count, and code size. This is the
// perf currency SPIR-V word count only loosely proxies — the driver re-optimizes
// SPIR-V, so only the emitted ISA tells you whether a change made a shader faster.
//
// Uses VK_KHR_pipeline_executable_properties (RADV/ACO exposes it) to read the
// statistics straight off the compiled pipeline — no RADV_DEBUG=asm scraping.
//
//   svsl_isa a.vert.spv b.frag.spv c.comp.spv ...
//
// Descriptor layout is reflected from each module so pipeline creation matches
// the shader's resource use; a fragment stage is paired with a trivial embedded
// vertex shader (a graphics pipeline needs a vertex stage). Dev-only tool: links
// libvulkan directly, unlike the zero-dependency core library.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

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

// --- .spv loading -----------------------------------------------------------

static uint32_t *load_spv(const char *path, size_t *out_words) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (size <= 0 || size % 4 != 0) { fclose(f); return NULL; }
	uint32_t *words = malloc((size_t)size);
	if (fread(words, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(words); return NULL; }
	fclose(f);
	*out_words = (size_t)size / 4;
	return words;
}

// --- SPIR-V reflection ------------------------------------------------------
// Just enough to build a compatible pipeline layout: the entry stage, each
// descriptor binding's (set, binding, type, count), and whether push constants
// are used. Two passes over the flat instruction stream, keyed by result id.

enum { TK_OTHER, TK_IMAGE, TK_SAMPLER, TK_SAMPLED_IMAGE, TK_STRUCT, TK_ARRAY, TK_RUNTIME_ARRAY };

typedef struct {
	uint8_t  kind;       // TK_*
	uint8_t  sampled;    // OpTypeImage: 1 = sampled, 2 = storage
	uint32_t elem;       // ARRAY/RUNTIME_ARRAY/POINTER element or pointee type id
	uint32_t len;        // ARRAY length (resolved constant)
	uint32_t ptr_class;  // OpTypePointer storage class (else 0xffffffff)
} type_info_t;

typedef struct {
	VkShaderStageFlagBits stage;
	VkDescriptorSetLayoutBinding bindings[64];
	uint32_t binding_count;
	int32_t  max_set;
	int      has_push;
} reflect_t;

static VkDescriptorType descriptor_type(const type_info_t *ti, uint32_t storage) {
	if (storage == 2)  return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;   // Uniform
	if (storage == 12) return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // StorageBuffer
	// UniformConstant (0): decide from the pointee type
	switch (ti->kind) {
	case TK_SAMPLER:       return VK_DESCRIPTOR_TYPE_SAMPLER;
	case TK_SAMPLED_IMAGE: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	case TK_IMAGE:         return ti->sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
	                                                : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	default:               return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // struct in UniformConstant → SSBO
	}
}

static int reflect(const uint32_t *w, size_t n, reflect_t *out) {
	if (n < 5 || w[0] != 0x07230203u) return 0;
	uint32_t bound = w[3];
	type_info_t *ti = calloc(bound, sizeof(type_info_t));
	int32_t     *deco_set  = malloc(bound * sizeof(int32_t));
	int32_t     *deco_bind = malloc(bound * sizeof(int32_t));
	uint32_t    *const_val = calloc(bound, sizeof(uint32_t));
	for (uint32_t i = 0; i < bound; i++) { deco_set[i] = -1; deco_bind[i] = -1; ti[i].ptr_class = 0xffffffffu; }

	memset(out, 0, sizeof(*out));
	out->max_set = -1;

	// pass 1: types, decorations, constants, entry stage
	for (size_t p = 5; p < n; ) {
		uint32_t word = w[p], len = word >> 16, op = word & 0xffff;
		if (len == 0 || p + len > n) break;
		const uint32_t *a = &w[p];
		switch (op) {
		case 15: // OpEntryPoint: a[1] = execution model
			out->stage = a[1] == 0 ? VK_SHADER_STAGE_VERTEX_BIT
			           : a[1] == 4 ? VK_SHADER_STAGE_FRAGMENT_BIT
			           : a[1] == 5 ? VK_SHADER_STAGE_COMPUTE_BIT : 0;
			break;
		case 71: // OpDecorate: a[1] target, a[2] decoration, a[3] operand
			if (a[2] == 34 && a[1] < bound) deco_set[a[1]]  = (int32_t)a[3]; // DescriptorSet
			if (a[2] == 33 && a[1] < bound) deco_bind[a[1]] = (int32_t)a[3]; // Binding
			break;
		case 43: if (a[2] < bound) const_val[a[2]] = a[3]; break;            // OpConstant (small)
		case 25: if (a[1] < bound) { ti[a[1]].kind = TK_IMAGE; ti[a[1]].sampled = (uint8_t)a[7]; } break;
		case 26: if (a[1] < bound)   ti[a[1]].kind = TK_SAMPLER; break;
		case 27: if (a[1] < bound) { ti[a[1]].kind = TK_SAMPLED_IMAGE; ti[a[1]].elem = a[2]; } break;
		case 28: if (a[1] < bound) { ti[a[1]].kind = TK_ARRAY; ti[a[1]].elem = a[2];
		                             ti[a[1]].len = a[3] < bound ? const_val[a[3]] : 1; } break;
		case 29: if (a[1] < bound) { ti[a[1]].kind = TK_RUNTIME_ARRAY; ti[a[1]].elem = a[2]; } break;
		case 30: if (a[1] < bound)   ti[a[1]].kind = TK_STRUCT; break;
		case 32: if (a[1] < bound) { ti[a[1]].ptr_class = a[2]; ti[a[1]].elem = a[3]; } break; // OpTypePointer
		default: break;
		}
		p += len;
	}

	// pass 2: variables → descriptor bindings
	for (size_t p = 5; p < n; ) {
		uint32_t word = w[p], len = word >> 16, op = word & 0xffff;
		if (len == 0 || p + len > n) break;
		const uint32_t *a = &w[p];
		if (op == 59) { // OpVariable: a[1] result type (a pointer), a[2] id, a[3] storage class
			uint32_t storage = a[3], var = a[2], ptr = a[1];
			if (storage == 9) { out->has_push = 1; }                       // PushConstant
			if ((storage == 0 || storage == 2 || storage == 12) && var < bound) {
				if (deco_set[var] < 0 || deco_bind[var] < 0) { p += len; continue; }
				uint32_t pointee = ptr < bound ? ti[ptr].elem : 0;
				uint32_t count   = 1;
				while (pointee < bound && (ti[pointee].kind == TK_ARRAY || ti[pointee].kind == TK_RUNTIME_ARRAY)) {
					count   = ti[pointee].kind == TK_ARRAY && ti[pointee].len ? ti[pointee].len : count;
					pointee = ti[pointee].elem;
				}
				if (out->binding_count < 64) {
					out->bindings[out->binding_count++] = (VkDescriptorSetLayoutBinding){
						.binding         = (uint32_t)deco_bind[var],
						.descriptorType  = descriptor_type(pointee < bound ? &ti[pointee] : &ti[0], storage),
						.descriptorCount = count ? count : 1,
						.stageFlags      = VK_SHADER_STAGE_ALL,
					};
					// stash the set index in a parallel slot via the high bits of a scratch:
					// simplest is a parallel array, but we re-derive set below.
				}
				if (deco_set[var] > out->max_set) out->max_set = deco_set[var];
			}
		}
		p += len;
	}

	// second sweep to record set indices alongside bindings (kept simple/robust)
	uint32_t bi = 0;
	int32_t  binding_set[64];
	for (size_t p = 5; p < n && bi < out->binding_count; ) {
		uint32_t word = w[p], len = word >> 16, op = word & 0xffff;
		if (len == 0 || p + len > n) break;
		const uint32_t *a = &w[p];
		if (op == 59) {
			uint32_t storage = a[3], var = a[2];
			if ((storage == 0 || storage == 2 || storage == 12) && var < bound &&
			    deco_set[var] >= 0 && deco_bind[var] >= 0)
				binding_set[bi++] = deco_set[var];
		}
		p += len;
	}
	// stash each binding's set index in the unused pImmutableSamplers slot; make_layout reads it back
	(void)bi;
	for (uint32_t i = 0; i < out->binding_count && i < 64; i++)
		out->bindings[i].pImmutableSamplers = (const VkSampler *)(uintptr_t)binding_set[i];

	free(ti); free(deco_set); free(deco_bind); free(const_val);
	return out->stage != 0;
}

// --- Vulkan setup -----------------------------------------------------------

static VkInstance       g_inst;
static VkPhysicalDevice g_phys;
static VkDevice         g_dev;
static PFN_vkGetPipelineExecutablePropertiesKHR            p_props;
static PFN_vkGetPipelineExecutableStatisticsKHR            p_stats;

static int vk_init(void) {
	VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
	VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
	if (vkCreateInstance(&ici, NULL, &g_inst) != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance failed\n"); return 0; }

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(g_inst, &count, NULL);
	VkPhysicalDevice devs[16];
	if (count > 16) count = 16;
	vkEnumeratePhysicalDevices(g_inst, &count, devs);
	g_phys = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < count; i++) {
		VkPhysicalDeviceProperties pr;
		vkGetPhysicalDeviceProperties(devs[i], &pr);
		if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
		    pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { g_phys = devs[i]; break; }
	}
	if (!g_phys) { fprintf(stderr, "no hardware GPU found\n"); return 0; }
	VkPhysicalDeviceProperties pr;
	vkGetPhysicalDeviceProperties(g_phys, &pr);
	fprintf(stderr, "device: %s\n", pr.deviceName);

	float prio = 1.0f;
	VkDeviceQueueCreateInfo q = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	                              .queueCount = 1, .pQueuePriorities = &prio };
	const char *ext[] = { VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME };
	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
		.pipelineExecutableInfo = VK_TRUE };
	VkPhysicalDeviceVulkan13Features v13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.dynamicRendering = VK_TRUE, .pNext = &pe };
	VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &v13,
	                           .queueCreateInfoCount = 1, .pQueueCreateInfos = &q,
	                           .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext };
	if (vkCreateDevice(g_phys, &dci, NULL, &g_dev) != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed\n"); return 0; }

	p_props = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutablePropertiesKHR");
	p_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutableStatisticsKHR");
	return p_props && p_stats;
}

static VkShaderModule make_module(const uint32_t *w, size_t words) {
	VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	                                .codeSize = words * 4, .pCode = w };
	VkShaderModule m = VK_NULL_HANDLE;
	vkCreateShaderModule(g_dev, &ci, NULL, &m);
	return m;
}

// Build a pipeline layout that satisfies the reflected bindings.
static VkPipelineLayout make_layout(const reflect_t *r, VkDescriptorSetLayout *sets, uint32_t *set_count) {
	int32_t max_set = r->max_set < 0 ? -1 : r->max_set;
	*set_count = (uint32_t)(max_set + 1);
	for (int32_t s = 0; s <= max_set; s++) {
		VkDescriptorSetLayoutBinding b[64];
		uint32_t bc = 0;
		for (uint32_t i = 0; i < r->binding_count; i++)
			if ((int32_t)(uintptr_t)r->bindings[i].pImmutableSamplers == s) {
				b[bc] = r->bindings[i];
				b[bc].pImmutableSamplers = NULL;
				bc++;
			}
		VkDescriptorSetLayoutCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		                                       .bindingCount = bc, .pBindings = b };
		vkCreateDescriptorSetLayout(g_dev, &ci, NULL, &sets[s]);
	}
	VkPushConstantRange pc = { .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 128 };
	VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	                                   .setLayoutCount = *set_count, .pSetLayouts = sets,
	                                   .pushConstantRangeCount = r->has_push ? 1 : 0, .pPushConstantRanges = &pc };
	VkPipelineLayout layout = VK_NULL_HANDLE;
	vkCreatePipelineLayout(g_dev, &lci, NULL, &layout);
	return layout;
}

// Query and print the statistics for the executable matching `stage`.
static void report(const char *file, VkPipeline pipe, VkShaderStageFlagBits stage) {
	uint32_t nexec = 0;
	VkPipelineInfoKHR pinfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe };
	p_props(g_dev, &pinfo, &nexec, NULL);
	VkPipelineExecutablePropertiesKHR *props = calloc(nexec, sizeof(*props));
	for (uint32_t i = 0; i < nexec; i++) props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
	p_props(g_dev, &pinfo, &nexec, props);

	for (uint32_t e = 0; e < nexec; e++) {
		if (!(props[e].stages & stage)) continue;
		VkPipelineExecutableInfoKHR ei = { .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
		                                   .pipeline = pipe, .executableIndex = e };
		uint32_t ns = 0;
		p_stats(g_dev, &ei, &ns, NULL);
		VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof(*st));
		for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
		p_stats(g_dev, &ei, &ns, st);

		long vgpr = -1, sgpr = -1, code = -1, insts = -1, spill_v = 0, spill_s = 0;
		for (uint32_t i = 0; i < ns; i++) {
			long v = (long)st[i].value.u64; // all RADV stats are u64/i64
			const char *nm = st[i].name;
			if      (strstr(nm, "Spilled VGPR")) spill_v = v;
			else if (strstr(nm, "Spilled SGPR")) spill_s = v;
			else if (strstr(nm, "VGPR"))         vgpr = v;
			else if (strstr(nm, "SGPR"))         sgpr = v;
			else if (strstr(nm, "Code size"))    code = v;
			else if (strstr(nm, "Instructions")) insts = v;
		}
		const char *sname = stage == VK_SHADER_STAGE_VERTEX_BIT ? "vertex"
		                  : stage == VK_SHADER_STAGE_FRAGMENT_BIT ? "fragment" : "compute";
		printf("ISA %-40s stage=%-8s vgpr=%-3ld sgpr=%-3ld insts=%-5ld code=%-5ld spill=%ld/%ld\n",
		       file, sname, vgpr, sgpr, insts, code, spill_v, spill_s);
		free(st);
		free(props);
		return;
	}
	printf("ISA %-40s (no executable for requested stage)\n", file);
	free(props);
}

static void measure(const char *file) {
	size_t words = 0;
	uint32_t *spv = load_spv(file, &words);
	if (!spv) { fprintf(stderr, "cannot read %s\n", file); return; }
	reflect_t r;
	if (!reflect(spv, words, &r)) { fprintf(stderr, "reflect failed %s\n", file); free(spv); return; }

	VkDescriptorSetLayout sets[16] = {0};
	uint32_t set_count = 0;
	VkPipelineLayout layout = make_layout(&r, sets, &set_count);
	VkShaderModule mod = make_module(spv, words);

	const VkPipelineCreateFlags CAP = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
	                                  VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
	VkPipeline pipe = VK_NULL_HANDLE;
	VkResult res;

	if (r.stage == VK_SHADER_STAGE_COMPUTE_BIT) {
		VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.flags = CAP, .layout = layout,
			.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			           .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "cs" } };
		res = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
	} else {
		int frag = r.stage == VK_SHADER_STAGE_FRAGMENT_BIT;
		// Pair a fragment with its sibling vertex shader (same basename, .vert.spv)
		// so interpolated inputs are defined — otherwise RADV treats them as
		// undefined and DCEs the dependent work, making the measurement useless.
		// Fall back to the trivial embedded VS only if no sibling is found.
		VkShaderModule vs_mod = mod;
		uint32_t *sib = NULL;
		if (frag) {
			char vspath[1024];
			snprintf(vspath, sizeof(vspath), "%s", file);
			char *dot = strstr(vspath, ".frag.spv");
			size_t sw = 0;
			if (dot) { memcpy(dot, ".vert.spv", 9); sib = load_spv(vspath, &sw); }
			vs_mod = sib ? make_module(sib, sw) : make_module(MIN_VS, sizeof(MIN_VS) / 4);
		}
		VkPipelineShaderStageCreateInfo stages[2]; uint32_t sc = 0;
		stages[sc++] = (VkPipelineShaderStageCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs_mod, .pName = "vs" };
		if (frag) stages[sc++] = (VkPipelineShaderStageCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mod, .pName = "ps" };

		VkPipelineVertexInputStateCreateInfo vin = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
		VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
		VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1, .scissorCount = 1 };
		VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.rasterizerDiscardEnable = frag ? VK_FALSE : VK_TRUE, .polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f };
		VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
		VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
		VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = frag ? 1 : 0, .pAttachments = &cba };
		VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo ds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = 2, .pDynamicStates = dyn };
		VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
		VkPipelineRenderingCreateInfo rend = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = frag ? 1 : 0, .pColorAttachmentFormats = &color_fmt };
		VkGraphicsPipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &rend, .flags = CAP, .stageCount = sc, .pStages = stages,
			.pVertexInputState = &vin, .pInputAssemblyState = &ia, .pViewportState = &vp,
			.pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb,
			.pDynamicState = &ds, .layout = layout };
		res = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
		if (frag) { vkDestroyShaderModule(g_dev, vs_mod, NULL); free(sib); }
	}

	if (res != VK_SUCCESS || !pipe) {
		printf("ISA %-40s (pipeline creation failed: %d)\n", file, res);
	} else {
		report(file, pipe, r.stage);
		vkDestroyPipeline(g_dev, pipe, NULL);
	}
	vkDestroyShaderModule(g_dev, mod, NULL);
	vkDestroyPipelineLayout(g_dev, layout, NULL);
	for (uint32_t s = 0; s < set_count; s++) vkDestroyDescriptorSetLayout(g_dev, sets[s], NULL);
	free(spv);
}

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: svsl_isa <file.spv>...\n"); return 2; }
	if (!vk_init()) return 1;
	for (int i = 1; i < argc; i++) measure(argv[i]);
	vkDestroyDevice(g_dev, NULL);
	vkDestroyInstance(g_inst, NULL);
	return 0;
}
