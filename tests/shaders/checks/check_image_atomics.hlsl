// Image atomics: every op must map to the right OpAtomic* (regression for the
// method-table aux vs signed_ops off-by-one that miscompiled all ops but Add).
Image2D<int,  r32i>  imgS;
Image2D<uint, r32ui> imgU;

[numthreads(1, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	int2 p = int2(id.xy);
	imgS.InterlockedAdd(p, 1);        // OpAtomicIAdd
	imgS.InterlockedMin(p, 5);        // OpAtomicSMin
	imgS.InterlockedMax(p, 7);        // OpAtomicSMax
	imgU.InterlockedMin(p, 5u);       // OpAtomicUMin
	imgU.InterlockedMax(p, 7u);       // OpAtomicUMax
	imgU.InterlockedAnd(p, 0xF0u);    // OpAtomicAnd
	imgU.InterlockedOr (p, 0x0Fu);    // OpAtomicOr
	imgU.InterlockedXor(p, 0xFFu);    // OpAtomicXor
	uint prev;
	imgU.InterlockedExchange(p, 42u, prev); // OpAtomicExchange

	// HLSL's subscript spelling reaches the same image-atomic path, and lowers
	// byte-identically to the method form above
	InterlockedAdd(imgS[p], 1);             // OpAtomicIAdd
	InterlockedMin(imgU[p], 5u);            // OpAtomicUMin
	uint prev2;
	InterlockedExchange(imgU[p], 42u, prev2); // OpAtomicExchange, out-param form
}
