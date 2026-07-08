//--name = check/enum

enum : int16 {
	One,
	Two
} option = One;

enum Mode : uint {
	ModeA = 1,
	ModeB = 4,
	ModeC,          // = 5
} mode;

enum { Red, Green, Blue };   // anonymous, no variable — just global constants

RWStructuredBuffer<uint> buf : register(u0);

[numthreads(1,1,1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	uint r = 0;
	if (option == Two)   r += 10;
	if (mode == ModeC)   r += 100;
	switch (mode) {
		case ModeA: r += 1; break;
		case ModeB: r += 2; break;
		default:    r += 3; break;
	}
	r += Green;               // enum const used as a plain int
	Mode m = ModeB;           // named enum as a type
	buf[tid.x] = r + (uint)m;
}
