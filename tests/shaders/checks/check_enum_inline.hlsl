//--name = check/enum_inline

// inline enum in a function parameter (the requested form)
uint doThing(enum { Thing1, Thing2, Thing3 } which_thing) {
	return (uint)which_thing * 2u;
}

// inline enum as a struct member type (packs as a raw int field)
struct Widget {
	enum : uint { Hidden, Shown } visibility : 1;
	uint id : 31;
};

RWStructuredBuffer<Widget> widgets : register(u0);

[numthreads(1,1,1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	// inline enum as a local variable type
	enum Dir { Up, Down, Left, Right } d = Right;

	widgets[tid.x].visibility = Shown;
	widgets[tid.x].id = doThing(Thing2) + (uint)d;
}
