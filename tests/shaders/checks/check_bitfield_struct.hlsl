//--name = check/bitfield_struct
// Packed struct bit fields (Step 2). Fields declare a resolve type on the left
// and a bit width (and optional normalized format) on the right of the colon:
//   uint  a : 4        raw unsigned, 4 bits
//   int   b : 6        raw signed, sign-extended on read
//   bool  f : 1        1-bit flag
//   float c : un10     unorm: [0,1] mapped over 10 bits (truncated on write)
//   half  d : sn6      snorm: [-1,1] mapped over 6 bits
//   half  h : 16       raw 16-bit half
// Fields pack densely LSB-first into backing uint32 words and never cross a word
// boundary (an oversized field auto-advances to the next word).

struct Packed {
	uint  a : 4;
	int   b : 6;
	bool  f : 1;
	float c : un10;
	half  d : sn6;
	half  h : 16;   // starts a new word (would cross the 32-bit boundary)
	uint  e;        // plain member, its own word
};

RWStructuredBuffer<Packed> buf : register(u0);

[numthreads(1, 1, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;

	// write every field kind
	buf[i].a = 13u;
	buf[i].b = -7;
	buf[i].f = true;
	buf[i].c = 0.75;
	buf[i].d = -0.5;
	buf[i].h = half(1.5);
	buf[i].e = 0xDEADBEEFu;

	// read them back and combine so nothing is dead-code eliminated
	uint  a = buf[i].a;
	int   b = buf[i].b;
	bool  f = buf[i].f;
	float c = buf[i].c;
	half  d = buf[i].d;
	half  h = buf[i].h;

	float sum = float(a) + float(b) + (f ? 1.0 : 0.0) + c + float(d) + float(h) + float(buf[i].e);
	buf[i].e = uint(sum);
}
