# APD run-time options

Every constant that upstream APDe-MVS hard-coded in the fusion, the weak-pixel filter, the coarse-to-fine
schedule and the PatchMatch kernels is a run-time option in this fork. All defaults equal the upstream
literals, so a run without options behaves like upstream.

## How to set options

- Command line: `APD --dense_folder <scene> --fusion.depth_mode=absolute --fusion.depth_abs=0.0002`
- INI file: `APD --dense_folder <scene> --config my.ini`, where the file holds `[group]` sections and
  `key = value` lines. Command-line values win over the file.
- Environment: `APD_CONFIG=<file>` is used when `--config` is absent (for wrappers that cannot pass arguments).
- Through `run.py`: `--apd_config my.ini` and/or repeated `--apd_opt group.key=value`.

APD prints the effective options at start-up and writes them to `<scene>/APD/apd_effective_config.ini`
(`apd_effective_config_fuse.ini` for `--only_fuse true`); that file can be passed back with `--config`.

## Schema for GUIs

`APD --dump_options_json` prints one JSON object per option: `name`, `group`, `key`, `type`
(`int`, `float`, `bool`, `enum`), `default`, optional `min`/`max`, optional `choices`, `unit`, `help`
and `stage`. `stage` is `fusion` when changing the option only needs `--only_fuse true` on existing depth
maps, and `patchmatch` when the depth maps must be recomputed.

## Notes

- Normals in `normals.bin` are already in the world frame, so the fusion normal test is valid for any
  angle between cameras; only its tolerance is an option.
- APD is not repeatable run to run, even with `pm.rng_seed` fixed, because the checkerboard kernels update
  shared state in GPU scheduling order. Compare settings over repeated runs.
- `views.*` options change the source-view lists and therefore also apply to `--only_fuse true`.

## Reference

| Option | Type | Default | Range / choices | Unit | Stage | Description |
|---|---|---|---|---|---|---|
| `camera.model` | enum | pinhole | pinhole, orthographic |  | patchmatch | Projection model of all cameras: pinhole, or orthographic for telecentric lenses written as long-focal-length pinhole equivalents (magnification = focal length / reference depth) |
| `camera.ortho_ref_depth` | float | 0 | 0 .. 1e+09 | world | patchmatch | Orthographic reference depth at which the pinhole-equivalent focal length is exact (0: per camera, the middle of its depth search range) |
| `fusion.depth_mode` | enum | relative | relative, absolute, pixel, off |  | fusion | Depth agreement test: relative (|dd|/d), absolute (world units), pixel (shift in the source image), off |
| `fusion.depth_rel` | float | 0.01 | 0 .. 1e+09 | ratio | fusion | Depth tolerance in relative mode |
| `fusion.depth_abs` | float | 0.1 | 0 .. 1e+09 | world | fusion | Depth tolerance in absolute mode, in the units of the camera translations |
| `fusion.depth_px` | float | 1 | 0 .. 1e+09 | px | fusion | Depth tolerance in pixel mode: allowed shift in the source image caused by the depth difference |
| `fusion.reproj_px` | float | 2 | 0 .. 1e+09 | px | fusion | Forward-backward reprojection tolerance in the reference image |
| `fusion.normal_test` | bool | true |  |  | fusion | Use normals in the consistency gate and score (normals are compared in the world frame) |
| `fusion.normal_max_rad` | float | 0.174533 | 0 .. 3.2 | rad | fusion | Largest normal angle between reference and source (0.174533 rad = 10 deg) |
| `fusion.score_w_reproj` | float | 1 | 0 .. 1e+09 | 1/px | fusion | Weight of the reprojection error in the consistency score exp(-sum) |
| `fusion.score_w_depth` | float | 200 | 0 .. 1e+09 |  | fusion | Weight of the relative depth error in the score (relative mode) |
| `fusion.score_w_depth_scaled` | float | 2 | 0 .. 1e+09 |  | fusion | Weight of depth error divided by its tolerance in the score (absolute and pixel modes) |
| `fusion.score_w_normal` | float | 10 | 0 .. 1e+09 | 1/rad | fusion | Weight of the normal angle in the score |
| `fusion.score_thresh_strong` | float | 0.3 | 0 .. 1 |  | fusion | Mean score a textured (STRONG) pixel must exceed to be fused |
| `fusion.score_thresh_weak` | float | 0.45 | 0 .. 1 |  | fusion | Mean score a low-texture (WEAK) pixel must exceed to be fused |
| `fusion.min_consistent` | int | 1 | 1 .. 32 | views | fusion | Minimum number of agreeing source views |
| `fusion.mask_used_pixels` | bool | true |  |  | fusion | Mark agreeing source pixels as consumed so they do not create duplicate points |
| `fusion.average_position` | bool | false |  |  | fusion | Fused point = mean of the reference point and the agreeing source points (false: reference point) |
| `fusion.view_min_angle_deg` | float | -1 | -1 .. 180 | deg | fusion | Skip source views whose optical axis is closer than this to the reference axis (<0: off) |
| `fusion.view_max_angle_deg` | float | -1 | -1 .. 180 | deg | fusion | Skip source views whose optical axis is further than this from the reference axis (<0: off) |
| `fusion.incident_max_deg` | float | -1 | -1 .. 180 | deg | fusion | A pixel takes part in fusion (as reference or source) only if the angle between its normal and the direction to its own camera is at most this (<0: off) |
| `fusion.incident_sigma_deg` | float | -1 | -1 .. 180 | deg | fusion | Soft incident-angle prior exp(-k^2/2s^2) on every source term and on the reference score (<0: off; Schoenberger et al. 2016 use 45) |
| `fusion.silhouette_trim_px` | int | 0 | 0 .. 100000 | px | fusion | Pixels within this many depth-map pixels of the view's silhouette edge (scene/sa_masks) take no part in fusion (0: off) |
| `fusion.occlusion_test` | bool | false |  |  | fusion | Behind-surface test: reject a point with at most occlusion_max_support agreeing sources when another view sees a surface more than occlusion_tol in front of it along its ray and that surface has at least occlusion_min_support agreeing sources (removes 2-view layers lying under a multi-view surface) |
| `fusion.occlusion_max_support` | int | 1 | 0 .. 32 | views | fusion | Reference pixels with at most this many agreeing sources are tested (1 = two-view points) |
| `fusion.occlusion_min_support` | int | 2 | 1 .. 32 | views | fusion | Agreeing sources an occluding pixel needs (2 = a surface seen consistently by at least three views) |
| `fusion.occlusion_tol` | float | 0.0001 | 0 .. 1e+09 | world | fusion | Distance along the occluding view's ray by which its surface must lie in front of the point |
| `fusion.occlusion_min_cos` | float | 0.2 | -1 .. 1 |  | fusion | A view takes part only if the point's normal and the occluding pixel's normal both face it with at least this cosine |
| `weakfilter.max_view_angle_deg` | float | 80 | 0 .. 180 | deg | fusion | Source views separated by more than this angle at the point are ignored |
| `weakfilter.depth_mode` | enum | relative | relative, absolute |  | fusion | Occlusion margin type: relative or absolute |
| `weakfilter.depth_rel` | float | 0.01 | 0 .. 1e+09 | ratio | fusion | Occlusion margin relative to the source depth |
| `weakfilter.depth_abs` | float | 0.1 | 0 .. 1e+09 | world | fusion | Occlusion margin in world units |
| `weakfilter.strong_occluded_min` | int | 2 | 1 .. 32 | views | fusion | Number of STRONG source pixels lying behind a WEAK point that removes it |
| `weakfilter.weak_occluded_min` | int | 4 | 1 .. 32 | views | fusion | Number of lower-confidence WEAK source pixels lying behind a WEAK point that removes it |
| `weakfilter.confidence_as_uchar` | bool | false |  |  | fusion | Read the confidence map with its stored type (false keeps the upstream float read) |
| `depth.range_scale_min` | float | 0.6 | 0 .. 1e+09 | ratio | patchmatch | Search range lower bound = DEPTH_MIN of the cam file times this (1 = no widening) |
| `depth.range_scale_max` | float | 1.2 | 0 .. 1e+09 | ratio | patchmatch | Search range upper bound = DEPTH_MAX of the cam file times this (1 = no widening) |
| `depth.range_pad` | float | 0 | -1e+09 .. 1e+09 | world | patchmatch | World units added to both ends of the scaled search range (negative shrinks it) |
| `views.all_pairs` | bool | false |  |  | patchmatch | Use every other image as source view; pair.txt only sets the order (scores are ignored) |
| `views.skip_self` | bool | false |  |  | patchmatch | Ignore pair.txt entries that name the reference image itself |
| `views.min_pair_score` | float | 0 | -1e+09 .. 1e+09 |  | patchmatch | pair.txt source views with a score at or below this are skipped |
| `views.max_src` | int | -1 | -1 .. 31 | views | patchmatch | Keep only the first N source views of every reference (<=0: all) |
| `views.min_angle_deg` | float | -1 | -1 .. 180 | deg | patchmatch | Drop source views whose optical axis is closer than this to the reference axis (<0: off) |
| `views.max_angle_deg` | float | -1 | -1 .. 180 | deg | patchmatch | Drop source views whose optical axis is further than this from the reference axis (<0: off) |
| `pipeline.round_max_size` | int | 800 | 16 .. 1e+06 | px | patchmatch | The image is halved until its longer side is at most this; sets the number of scales |
| `pipeline.rounds` | int | -1 | -1 .. 16 |  | patchmatch | Force the number of scales (<=0: derive from round_max_size) |
| `pipeline.extra_rounds` | int | 0 | 0 .. 8 |  | patchmatch | Scales added to the automatic count (each one starts at half the size of the previous coarsest; ignored when rounds > 0; never shrinks the image below 32 px) |
| `pipeline.geom_iterations` | int | 3 | 0 .. 64 |  | patchmatch | Geometric-consistency passes per scale |
| `pipeline.max_iterations` | int | 3 | 1 .. 64 |  | patchmatch | Checkerboard propagation iterations per pass |
| `pipeline.init_weak_peak_radius` | int | 6 | 0 .. 30 | steps | patchmatch | Cost-curve peak tolerance of the first pass of each scale |
| `pipeline.weak_peak_start` | int | 4 | 0 .. 30 | steps | patchmatch | Peak tolerance of geometric pass j = max(start - step * j, min) |
| `pipeline.weak_peak_step` | int | 2 | 0 .. 30 | steps | patchmatch | See weak_peak_start |
| `pipeline.weak_peak_min` | int | 2 | 0 .. 30 | steps | patchmatch | See weak_peak_start |
| `pipeline.ransac_base` | float | 0.01 | 0 .. 1e+09 | ratio | patchmatch | Anchor plane inlier threshold = base - scale_index * step (fraction of the depth range) |
| `pipeline.ransac_step` | float | 0.00125 | 0 .. 1e+09 | ratio | patchmatch | See ransac_base |
| `pipeline.rotate_time_max` | int | 4 | 1 .. 4 |  | patchmatch | Anchor search rotations per direction = min(2^scale_index, this) |
| `pm.strong_radius` | int | 5 | 1 .. 64 | px | patchmatch | NCC window radius of textured pixels and of the centre patch |
| `pm.strong_increment` | int | 2 | 1 .. 64 | px | patchmatch | Sampling step inside the strong NCC window |
| `pm.weak_radius` | int | 5 | 1 .. 64 | px | patchmatch | NCC window radius around each anchor of a low-texture pixel |
| `pm.weak_increment` | int | 5 | 1 .. 64 | px | patchmatch | Sampling step inside the anchor NCC window |
| `pm.top_k` | int | 4 | 1 .. 31 | views | patchmatch | Number of best source views used for the initial cost and view selection |
| `pm.geom_factor` | float | -1 | -1 .. 1e+09 | 1/px | patchmatch | Weight of the geometric consistency cost (<0: dataset default, 0.2 or 0.05 for TaT) |
| `pm.geom_max_cost` | float | 3 | 0 .. 1e+09 | px | patchmatch | Cap of the geometric consistency (forward-backward reprojection) cost |
| `pm.depth_perturbation` | float | 0.02 | 0 .. 1 | ratio | patchmatch | Relative depth perturbation of the local refinement hypotheses |
| `pm.normal_perturbation` | float | 0.02 | 0 .. 1 | pi | patchmatch | Normal perturbation of the local refinement hypotheses, as a fraction of pi |
| `pm.refine_init_margin` | float | 0.1 | 0 .. 2 | cost | patchmatch | Cost improvement needed to replace an upsampled hypothesis in the first pass of a scale |
| `pm.weak_center_weight` | float | 0.25 | 0 .. 1 |  | patchmatch | Deformable NCC = w * centre patch + (1 - w) * anchor patches |
| `pm.median_filter` | bool | true |  |  | patchmatch | Median-filter the depth of textured pixels after propagation |
| `pm.local_refine` | bool | true |  |  | patchmatch | Final per-pixel depth refinement over disparity steps |
| `pm.local_refine_radius` | int | 5 | 0 .. 64 | steps | patchmatch | Search radius of the final refinement |
| `pm.local_refine_margin` | float | 0.1 | 0 .. 2 | cost | patchmatch | Cost improvement needed to accept the refined depth |
| `pm.rng_seed` | int | -1 | -1 .. 9e+18 |  | patchmatch | Random seed (<0: seed from the clock, runs are not repeatable) |
| `pm.depth_perturbation_mode` | enum | relative | relative, range, absolute |  | patchmatch | Depth perturbation window: relative = +-depth_perturbation * depth (upstream; about +-60 mm at a 3 m pinhole-equivalent depth), range = +-depth_perturbation * width of the pixel's search range, absolute = +-depth_perturbation_abs; range and absolute are clipped to the search range |
| `pm.depth_perturbation_abs` | float | 0 | 0 .. 1e+09 | world | patchmatch | Half-width of the absolute depth perturbation (depth_perturbation_mode = absolute) |
| `pm.perturbation_clip` | bool | false |  |  | patchmatch | Draw the perturbed depth inside the pixel's search range (upstream's retry loop never retries); always on for the range/absolute modes and with a depth prior |
| `prior.enable` | bool | false |  |  | patchmatch | Read <dense>/depth_prior/<ref>.bin (BinMat, full image size; float depth or float2 lo/hi, 0 = none) and confine init, refinement, perturbation and propagation of each pixel to its band; pixels without a prior keep the view range. Prior depths are absolute (not scaled by depth.range_scale_*) |
| `prior.band_near` | float | 0.0005 | 0 .. 1e+09 | world | patchmatch | Band towards the camera around a 1-channel prior depth: lo = prior - this |
| `prior.band_far` | float | 0.0005 | 0 .. 1e+09 | world | patchmatch | Band away from the camera around a 1-channel prior depth: hi = prior + this |
| `viewsel.prior_selected` | float | 0.9 | 0 .. 1 |  | patchmatch | Prior of a view that a neighbouring pixel selected |
| `viewsel.prior_unselected` | float | 0.1 | 0 .. 1 |  | patchmatch | Prior of a view that a neighbouring pixel did not select |
| `viewsel.cost_thresh_init` | float | 0.8 | 0 .. 2 | cost | patchmatch | Good-cost threshold = init * exp(-iteration^2 / decay) |
| `viewsel.cost_thresh_decay` | float | 90 | 1e-06 .. 1e+09 |  | patchmatch | See cost_thresh_init |
| `viewsel.good_sigma` | float | 0.18 | 1e-06 .. 1e+09 |  | patchmatch | View weight of a good cost = exp(-cost^2 / sigma) |
| `viewsel.fallback_sigma` | float | 0.32 | 1e-06 .. 1e+09 |  | patchmatch | View weight when too few costs are good = exp(-threshold^2 / sigma) |
| `viewsel.bad_cost` | float | 1.2 | 0 .. 2 | cost | patchmatch | NCC cost above which a hypothesis counts as bad for a view |
| `viewsel.min_good` | int | 2 | 0 .. 8 |  | patchmatch | A view needs more than this many good hypotheses (of 8) |
| `viewsel.max_bad` | int | 3 | 0 .. 9 |  | patchmatch | A view is rejected at this many bad hypotheses (of 8) |
| `viewsel.num_samples` | int | 15 | 1 .. 255 |  | patchmatch | Monte-Carlo view samples drawn per pixel |
| `weak.border_margin` | int | 6 | 0 .. 1024 | px | patchmatch | Image border that is never classified and never used as anchor |
| `weak.max_peak_cost` | float | 0.5 | 0 .. 2 | cost | patchmatch | A cost-curve minimum above this makes the pixel WEAK |
| `weak.strong_single_peak_cost` | float | 0.15 | 0 .. 2 | cost | patchmatch | A single-minimum cost curve below this makes the pixel STRONG |
| `weak.strong_multi_peak_var` | float | 0.2 | 0 .. 2 | cost | patchmatch | A multi-minimum cost curve with a spread above this makes the pixel STRONG |
| `weak.filter_radius` | int | 2 | 0 .. 64 | px | patchmatch | A STRONG pixel without another STRONG pixel in this radius becomes UNKNOWN |
| `weak.nearest_strong_radius` | int | 100 | 1 .. 1024 | px | patchmatch | Search radius for the nearest STRONG pixel (cost grows with the square) |
| `weak.anchor_ransac_iters` | int | 50 | 1 .. 100000 |  | patchmatch | RANSAC iterations when choosing the anchors of a WEAK pixel |
| `weak.anchor_min_inliers` | int | 6 | 3 .. 32 |  | patchmatch | Minimum inliers of the anchor plane |
| `weak.fit_ransac_iters` | int | 50 | 1 .. 100000 |  | patchmatch | RANSAC iterations of the per-iteration plane fit of a WEAK pixel |
| `conf.reproj_px` | float | 2 | 0 .. 1e+09 | px | patchmatch | Reprojection tolerance of the confidence vote |
| `conf.depth_rel` | float | 0.02 | 0 .. 1e+09 | ratio | patchmatch | Relative depth tolerance of the confidence vote |
| `conf.w_exist` | int | 1 | 0 .. 8 | votes | patchmatch | Votes for a source view that has a depth at the projected pixel |
| `conf.w_reproj` | int | 2 | 0 .. 8 | votes | patchmatch | Votes for passing the reprojection test |
| `conf.w_depth` | int | 2 | 0 .. 8 | votes | patchmatch | Votes for passing the depth test |
