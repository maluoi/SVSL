#!/usr/bin/env bash
# Refreshes the pinned StereoKit shader corpus copies in this directory.
# Corpus shaders are copied (not referenced) so upstream edits don't break tests.
#
# ported/ is NOT refreshed: those started as a one-time import from the frozen
# ~/projects/spirv_sl prototype and are maintained here (lightly ported to v1
# syntax where the prototype used its own pre-v1 spellings).
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"
sk="$HOME/SK/StereoKit"
mw="$HOME/repositories/SKMorrowind/Assets"

mkdir -p "$dir/include" "$dir/builtin" "$dir/examples" "$dir/morrowind"
cp "$sk"/tools/include/*.hlsli                  "$dir/include/"
cp "$sk"/StereoKitC/shaders_builtin/*.hlsl      "$dir/builtin/"
cp "$sk"/Examples/Assets/Shaders/*.hlsl         "$dir/examples/"
cp "$sk"/Examples/StereoKitCTest/Shaders/*.hlsl "$dir/examples/"
# SKMorrowind: heavy real-world shaders (Texture3D GI, comparison samplers,
# storage image formats, discard); their .hlsli includes live alongside them
cp "$mw"/*.hlsl "$mw"/*.hlsli                   "$dir/morrowind/"

# strip the legacy never-taken struct-inheritance clause (unsupported by SVSL,
# and sk_ps_input_t is defined nowhere upstream either)
sed -i 's/ : sk_ps_input_t//' "$dir"/examples/*.hlsl

echo "corpus refreshed:"
ls "$dir/include" "$dir/builtin" "$dir/examples" "$dir/morrowind" | grep -c hlsl || true
