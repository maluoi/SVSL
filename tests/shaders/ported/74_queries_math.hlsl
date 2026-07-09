// Tier 1-3 SPIR-V coverage: buffer GetDimensions (OpArrayLength + stride
// constant), wave ballot bit counts, InterlockedCompareStore, out-parameter
// math (sincos/modf/frexp), rcp/ldexp/degrees/radians, the GetDimensions
// mip-level overload (OpImageQuerySizeLod + OpImageQueryLevels), and
// CalculateLevelOfDetail (OpImageQueryLod).

struct item_t { float4 pos; float2 uv; };
std430 StructuredBuffer<item_t>   items;
RWStructuredBuffer<float4> outp;
Texture2D    tex;
SamplerState tex_s;

groupshared uint flag;

float4 out_param_math(float2 uv) {
	float a = rcp(uv.x) + degrees(uv.y) + radians(uv.x);
	float b = ldexp(uv.x, uv.y);
	float s, c;
	sincos(a, s, c);
	float ip;
	float fr = modf(b, ip);
	float ex;
	float mant = frexp(a, ex);
	return float4(s + c, fr + ip, mant + ex, 1);
}

float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
	uint w, h, levels;
	tex.GetDimensions(1, w, h, levels);
	float lod  = tex.CalculateLevelOfDetail(tex_s, uv);
	float lodu = tex.CalculateLevelOfDetailUnclamped(tex_s, uv);
	return out_param_math(uv) + tex.Sample(tex_s, uv) * (lod + lodu + (float)(w + h + levels));
}

[numthreads(64,1,1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint count, stride;
	items.GetDimensions(count, stride);
	uint just_count = items.GetDimensions();
	uint n      = WaveActiveCountBits(id.x % 2 == 0);
	uint before = WavePrefixCountBits(id.x % 3 == 0);
	InterlockedCompareStore(flag, 0, id.x);
	outp[id.x] = items[id.x % count].pos * (float)(n + before + stride + just_count + flag);
}
