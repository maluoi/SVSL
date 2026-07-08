//--name = check_io_locations
// Regression test for stage-IO location assignment. Each interface member must
// consume the right number of locations: an array spans one per element, a
// matrix one per row/column vector. When members only claimed a single location
// each, an array or matrix varying overlapped the members that followed it and
// spirv-val rejected the module ("conflicting input location assignment").
// Here colors[3] must occupy locations 0-2, basis (float3x3) 3-5, and uv 6.

struct vsIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
struct psIn {
	float4   pos       : SV_POSITION;
	float4   colors[3] : COLOR0;      // locations 0,1,2
	float3x3 basis     : TEXCOORD0;   // locations 3,4,5
	float2   uv        : TEXCOORD3;   // must land at location 6
};

psIn vs(vsIn input) {
	psIn o;
	o.pos       = input.pos;
	o.colors[0] = float4(input.uv, 0, 1);
	o.colors[1] = float4(1, 0, 0, 1);
	o.colors[2] = float4(0, 1, 0, 1);
	o.basis     = float3x3(1,0,0, 0,1,0, 0,0,1);
	o.uv        = input.uv;
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	return input.colors[0] + input.colors[1] + input.colors[2]
	     + float4(mul(input.basis, float3(input.uv, 1)), 1);
}
