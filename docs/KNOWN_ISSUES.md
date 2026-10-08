# Known issues

Open points in opt-in options. None of them affects a run with default options.

## fusion.occlusion_test (behind-surface test; off by default)

Found in the independent review of the merge of `feat/deep-layer-occlusion-test`; deferred because they
only matter with `fusion.occlusion_test=true`.

- `fusion.occlusion_tol` is an absolute tolerance (default 1e-4 world units), so it also catches
  pixel-rounding and depth noise. A relative tolerance, or one tied to `fusion.depth_rel`, would be better.
- Occluders ignore `masks[k]`, `skip_weaks[k]` and the reference's `src_allowed` angle window, and the
  support count in the pre-pass is looser than fusion's acceptance (no consistency-score threshold).
- Latent data race: `support_task` calls `imageIdToindexMap[...]` (`operator[]`, which may insert) from
  worker threads. It should use `.at()`.
- Pre-existing: the depth/normal loader's size-mismatch `continue` can leave `depths` shorter than
  `num_images`, which the per-view loops then index past.
- Wording: "along its ray" in the option help means the z-depth in the occluding view, not the
  Euclidean distance along the ray.
- `RunFusion_TAT_I` and `RunFusion_TAT_A` (Tanks and Temples fusion paths) ignore this option.
