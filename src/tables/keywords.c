#include "keywords.h"

#include <assert.h>

typedef struct keyword_row_t {
	const char *name;
	svsl_kw_    kw;
} keyword_row_t;

static const keyword_row_t keyword_table[] = {
	{ "break",           svsl_kw_break            },
	{ "case",            svsl_kw_case             },
	{ "cbuffer",         svsl_kw_cbuffer          },
	{ "centroid",        svsl_kw_centroid         },
	{ "coherent",        svsl_kw_coherent         },
	{ "const",           svsl_kw_const            },
	{ "continue",        svsl_kw_continue         },
	{ "default",         svsl_kw_default          },
	{ "demote",          svsl_kw_demote           },
	{ "discard",         svsl_kw_discard          },
	{ "do",              svsl_kw_do               },
	{ "else",            svsl_kw_else             },
	{ "enum",            svsl_kw_enum             },
	{ "false",           svsl_kw_false            },
	{ "flat",            svsl_kw_flat             },
	{ "for",             svsl_kw_for              },
	{ "groupshared",     svsl_kw_groupshared      },
	{ "if",              svsl_kw_if               },
	{ "in",              svsl_kw_in               },
	{ "include",         svsl_kw_include          },
	{ "inout",           svsl_kw_inout            },
	{ "invariant",       svsl_kw_invariant        },
	{ "nointerpolation", svsl_kw_nointerpolation  },
	{ "noperspective",   svsl_kw_noperspective    },
	{ "out",             svsl_kw_out              },
	{ "pack1",           svsl_kw_pack1            },
	{ "pack16",          svsl_kw_pack16           },
	{ "pack8",           svsl_kw_pack8            },
	{ "precise",         svsl_kw_precise          },
	{ "pushconstant",    svsl_kw_pushconstant     },
	{ "readonly",        svsl_kw_readonly         },
	{ "register",        svsl_kw_register         },
	{ "return",          svsl_kw_return           },
	{ "sample",          svsl_kw_sample           },
	{ "specialization",  svsl_kw_specialization   },
	{ "spirv",           svsl_kw_spirv            },
	{ "spirv_asm",       svsl_kw_spirv            },
	{ "static",          svsl_kw_static           },
	{ "storagebuffer",   svsl_kw_storagebuffer    },
	{ "struct",          svsl_kw_struct           },
	{ "switch",          svsl_kw_switch           },
	{ "true",            svsl_kw_true             },
	{ "uniform",         svsl_kw_uniform          },
	{ "volatile",        svsl_kw_volatile         },
	{ "while",           svsl_kw_while            },
	{ "workgroup",       svsl_kw_workgroup        },
	{ "writeonly",       svsl_kw_writeonly        },
};

static uint32_t keyword_hash(svsl_str_t s) {
	uint32_t c0 = (uint8_t)s.ptr[0], c1 = s.len > 1 ? (uint8_t)s.ptr[1] : 0, cn = (uint8_t)s.ptr[s.len - 1];
	return ((uint32_t)s.len * 131u + c0 * 31u + c1 * 17u + cn * 7u) & (SVSL_KEYWORD_SLOTS - 1);
}

void svsl_keyword_index_build(svsl_keyword_index_t *out) {
	*out = (svsl_keyword_index_t){0};
	for (int32_t i = 0; i < (int32_t)(sizeof(keyword_table) / sizeof(keyword_table[0])); i++) {
		uint32_t h = keyword_hash(svsl_str(keyword_table[i].name));
		while (out->slot[h]) h = (h + 1) & (SVSL_KEYWORD_SLOTS - 1);
		out->slot[h] = (uint8_t)(i + 1);
	}
#ifndef NDEBUG
	for (int32_t i = 0; i < (int32_t)(sizeof(keyword_table) / sizeof(keyword_table[0])); i++)
		assert(svsl_keyword_find(out, svsl_str(keyword_table[i].name)) == keyword_table[i].kw);
#endif
}

svsl_kw_ svsl_keyword_find(const svsl_keyword_index_t *index, svsl_str_t ident) {
	if (ident.len == 0) return svsl_kw_none;
	for (uint32_t h = keyword_hash(ident);; h = (h + 1) & (SVSL_KEYWORD_SLOTS - 1)) {
		uint8_t row = index->slot[h];
		if (!row) return svsl_kw_none;
		if (svsl_str_eq_cstr(ident, keyword_table[row - 1].name)) return keyword_table[row - 1].kw;
	}
}
