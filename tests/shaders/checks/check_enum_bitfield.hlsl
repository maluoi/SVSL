//--name = check/enum_bitfield
enum Facing : uint { North, East, South, West };  // 0..3, fits in 2 bits

struct Tile {
	Facing dir : 2;    // enum-typed bit field → raw 2-bit uint
	uint   hp  : 6;
	bool   solid : 1;
};

RWStructuredBuffer<Tile> tiles : register(u0);

[numthreads(1,1,1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	tiles[tid.x].dir   = West;   // 3
	tiles[tid.x].hp    = 42u;
	tiles[tid.x].solid = true;
	Facing f = tiles[tid.x].dir;
	if (f == West) tiles[tid.x].hp = 63u;
}
