# Known issues

Open points found while working on the fusion and PatchMatch input code.

Fixed on 2026-10-08 and no longer listed: absolute `fusion.occlusion_tol`; occluders ignoring masks,
see-through pixels and the angle window; the `operator[]` race in the occlusion support pre-pass; unknown
source ids read as view 0; the short per-view vectors after a size mismatch; the "along its ray" wording; the
silent TaT handling of `fusion.*`; `WeakVisFilter` reading the 8-bit confidence maps as float (past the end of
the matrix); the TaT per-source comparison values not being reset per pixel; the PatchMatch input stage
ignoring read failures (it now stops with an error naming the file).

Reproducing pre-fix output: the weak-filter confidence fix can change default fusion output when the weak
filter drops pixels (on the golden and mouse1 ctrl scenes it does not; see commit 7713bbf). Binaries with the
old float read are kept in `D:\Reconstruction\build\apde-mvs-custom-pre-master-80476d0` (80476d0, before all
of these fixes) and `D:\Reconstruction\build\apde-mvs-custom-pre-confidence-7f96e11` (7f96e11, all fusion
fixes except the confidence read, the TaT reset and the PatchMatch input checks).

## Tanks and Temples fusion (`--dataset TaT_a` / `TaT_i`)

- Both variants ignore `fusion.*` and `weakfilter.*` (they keep upstream's hard-coded thresholds and call
  `WeakVisFilter` with its defaults). APD warns for every such option set to a non-default value.

## fusion.occlusion_test (off by default): `occlusion_tol` on the mouse1 ctrl scene

Scored with the v19deeplayer scoring (`score19.py`: v10sparse score, v13 eval, deep17, circ17, region
coverage) after Recova's silhouette filter, binary 88fe833, v19 fusion options. "off" is byte-identical to
v19's control F0. Old = the pre-fix test (v19deeplayer SUMMARY, branch build 2e9f68e, whose
output equals 2b89955; `occlusion_tol` 1e-4).

| Variant | Points | Completeness / hole mm2 | Coverage ventricles / atria | Recovered of clean2 loss | Structure pts | Deep / main layer |
|---|---|---|---|---|---|---|
| off | 1145989 | 0.931 / 12.02 | 0.8296 / 0.4098 | 100 % | 41463 | 970 / 3876 |
| support >= 2, tol 0 (default) | 1116564 | 0.930 / 12.36 | 0.8293 / 0.4080 | 99.7 % | 34815 | 684 / 3673 |
| support >= 2, tol 1e-4 | 1136929 | 0.930 / 12.36 | 0.8293 / 0.4080 | 99.7 % | 36528 | 687 / 3799 |
| support >= 1, tol 0 | 1098352 | 0.929 / 12.42 | 0.8289 / 0.4072 | 99.2 % | 30632 | 433 / 3558 |
| support >= 1, tol 1e-4 | 1126216 | 0.929 / 12.42 | 0.8290 / 0.4072 | 99.2 % | 33376 | 441 / 3748 |
| old, support >= 2 | - | 0.926 / 13.0 | 0.829 / 0.407 | 99.6 % | 34.4k | 625 / 3771 |
| old, support >= 1 | - | 0.920 / 14.6 | 0.828 / 0.404 | 98.6 % | 28.5k | 290 / 3687 |

- Open: on this scene `occlusion_tol = 1e-4` (0.1 mm) removes the same deep layer as the scale-free default
  0 (687 vs 684 points, support >= 2; 441 vs 433, support >= 1) but keeps 126-190 more main-layer points and
  20-28k more points overall, with equal completeness and coverage. The default 0 is kept because it does not
  depend on the scene unit; for mouse1-type scenes (metres, ~15 um pixels) 1e-4 is the better setting.
- The fixed test removes less of the deep layer than the old one (support >= 1: 441 vs 290), because only
  pixels that fusion would accept count as occluders now, and keeps completeness and the hole area closer to
  "off". Whether the cloud or the mesh is better end to end (checkpoint chain, NCC, bumps) is not evaluated.
