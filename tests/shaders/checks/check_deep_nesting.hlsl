// Control-flow nesting deeper than the old fixed cf[32] emit stack (and the
// if/else-matching stack[64]). Regression for the unbounded-nesting overflow —
// the CF stacks are arena-sized to the instruction count now.
float4 ps(float4 pos : SV_Position) : SV_Target {
	float r = 0;
	int k = (int)pos.x;
	if (k > 0) {
		r += 0.0;
		if (k > 1) {
			r += 1.0;
			if (k > 2) {
				r += 2.0;
				if (k > 3) {
					r += 3.0;
					if (k > 4) {
						r += 4.0;
						if (k > 5) {
							r += 5.0;
							if (k > 6) {
								r += 6.0;
								if (k > 7) {
									r += 7.0;
									if (k > 8) {
										r += 8.0;
										if (k > 9) {
											r += 9.0;
											if (k > 10) {
												r += 10.0;
												if (k > 11) {
													r += 11.0;
													if (k > 12) {
														r += 12.0;
														if (k > 13) {
															r += 13.0;
															if (k > 14) {
																r += 14.0;
																if (k > 15) {
																	r += 15.0;
																	if (k > 16) {
																		r += 16.0;
																		if (k > 17) {
																			r += 17.0;
																			if (k > 18) {
																				r += 18.0;
																				if (k > 19) {
																					r += 19.0;
																					if (k > 20) {
																						r += 20.0;
																						if (k > 21) {
																							r += 21.0;
																							if (k > 22) {
																								r += 22.0;
																								if (k > 23) {
																									r += 23.0;
																									if (k > 24) {
																										r += 24.0;
																										if (k > 25) {
																											r += 25.0;
																											if (k > 26) {
																												r += 26.0;
																												if (k > 27) {
																													r += 27.0;
																													if (k > 28) {
																														r += 28.0;
																														if (k > 29) {
																															r += 29.0;
																															if (k > 30) {
																																r += 30.0;
																																if (k > 31) {
																																	r += 31.0;
																																	if (k > 32) {
																																		r += 32.0;
																																		if (k > 33) {
																																			r += 33.0;
																																			if (k > 34) {
																																				r += 34.0;
																																				if (k > 35) {
																																					r += 35.0;
																																					if (k > 36) {
																																						r += 36.0;
																																						if (k > 37) {
																																							r += 37.0;
																																							if (k > 38) {
																																								r += 38.0;
																																								if (k > 39) {
																																									r += 39.0;
																																								}
																																							}
																																						}
																																					}
																																				}
																																			}
																																		}
																																	}
																																}
																															}
																														}
																													}
																												}
																											}
																										}
																									}
																								}
																							}
																						}
																					}
																				}
																			}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return r.xxxx;
}
