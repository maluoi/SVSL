#include "semantics.h"

#include "../../vendor/spirv.h"

static bool ieq_prefix(svsl_str_t s, const char *upper, int32_t *out_digit) {
	int32_t i = 0;
	for (; upper[i]; i++) {
		if (i >= s.len) return false;
		char c = s.ptr[i];
		if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
		if (c != upper[i]) return false;
	}
	if (i == s.len) { *out_digit = 0; return true; }
	// optional trailing index digits (TEXCOORD3, SV_Target2)
	int32_t v = 0;
	for (; i < s.len; i++) {
		if (s.ptr[i] < '0' || s.ptr[i] > '9') return false;
		v = v * 10 + (s.ptr[i] - '0');
	}
	*out_digit = v;
	return true;
}

bool svsl_semantic_lookup(svsl_str_t semantic, svsl_sem_io_ io, svsl_semantic_info_t *out_info) {
	*out_info = (svsl_semantic_info_t){0};
	if (semantic.len == 0) return false;
	int32_t n = 0;

	if (ieq_prefix(semantic, "SV_POSITION", &n)) {
		if (io == svsl_sem_vs_in) return false; // vertex attribute, numbered like the rest
		out_info->is_builtin = true;
		out_info->builtin    = io == svsl_sem_ps_in ? SpvBuiltInFragCoord : SpvBuiltInPosition;
		return true;
	}
	if (ieq_prefix(semantic, "SV_TARGET", &n) && io == svsl_sem_ps_out) {
		out_info->target_index = n;
		return true; // location n, not a builtin
	}
	if (ieq_prefix(semantic, "SV_DEPTHGREATEREQUAL", &n)) {
		out_info->is_builtin = true;
		out_info->builtin    = SpvBuiltInFragDepth;
		out_info->depth_mode = 1;
		return true;
	}
	if (ieq_prefix(semantic, "SV_DEPTHLESSEQUAL", &n)) {
		out_info->is_builtin = true;
		out_info->builtin    = SpvBuiltInFragDepth;
		out_info->depth_mode = 2;
		return true;
	}
	if (ieq_prefix(semantic, "SV_DEPTH", &n)) {
		out_info->is_builtin = true;
		out_info->builtin    = SpvBuiltInFragDepth;
		return true;
	}
	// builtin only in the right stage/direction; elsewhere (SV_VertexID passed
	// VS→PS as a varying) these are plain numbered IO
	struct row { const char *name; SpvBuiltIn builtin; uint8_t io_mask; };
	#define IO(x) (1u << (x))
	static const struct row rows[] = {
		{ "SV_VERTEXID",         SpvBuiltInVertexIndex,          IO(svsl_sem_vs_in) },
		{ "SV_INSTANCEID",       SpvBuiltInInstanceIndex,        IO(svsl_sem_vs_in) },
		{ "SV_VIEWID",           SpvBuiltInViewIndex,            IO(svsl_sem_vs_in) | IO(svsl_sem_ps_in) },
		{ "SV_ISFRONTFACE",      SpvBuiltInFrontFacing,          IO(svsl_sem_ps_in) },
		{ "SV_SAMPLEINDEX",      SpvBuiltInSampleId,             IO(svsl_sem_ps_in) },
		{ "SV_COVERAGE",         SpvBuiltInSampleMask,           IO(svsl_sem_ps_in) | IO(svsl_sem_ps_out) },
		{ "SV_PRIMITIVEID",      SpvBuiltInPrimitiveId,          IO(svsl_sem_ps_in) },
		{ "SV_DISPATCHTHREADID", SpvBuiltInGlobalInvocationId,   IO(svsl_sem_cs_in) },
		{ "SV_GROUPTHREADID",    SpvBuiltInLocalInvocationId,    IO(svsl_sem_cs_in) },
		{ "SV_GROUPID",          SpvBuiltInWorkgroupId,          IO(svsl_sem_cs_in) },
		{ "SV_GROUPINDEX",       SpvBuiltInLocalInvocationIndex, IO(svsl_sem_cs_in) },
	};
	#undef IO
	for (int32_t i = 0; i < (int32_t)(sizeof(rows) / sizeof(rows[0])); i++) {
		if (ieq_prefix(semantic, rows[i].name, &n)) {
			if (!(rows[i].io_mask & (1u << io))) return false;
			out_info->is_builtin = true;
			out_info->builtin    = rows[i].builtin;
			return true;
		}
	}
	return false; // NORMAL0, TEXCOORD2, COLOR1, … — plain numbered IO
}
