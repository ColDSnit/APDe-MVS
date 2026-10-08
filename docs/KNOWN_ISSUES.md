# Known issues

Open points found while working on the fusion code. The items fixed on 2026-10-08 (absolute
`fusion.occlusion_tol`, occluders ignoring masks / see-through pixels / the angle window, the `operator[]`
race in the occlusion support pre-pass, unknown source ids read as view 0, the short per-view vectors after a
size mismatch, the "along its ray" wording, the silent TaT handling of `fusion.*`) are no longer listed.

## Default path (pre-existing, upstream behaviour kept for byte-identical output)

- `WeakVisFilter` reads the `CV_8UC1` confidence maps with `at<float>` unless
  `weakfilter.confidence_as_uchar = true`. Besides comparing reinterpreted bytes, this reads past the end of
  the matrix for WEAK pixels in roughly the last three rows (`4 * c` bytes into a row of `cols` bytes), so the
  comparison there uses whatever lies after the buffer. Fixing it changes default output, so it is left to
  an explicit decision; `confidence_as_uchar = true` avoids it.

## Tanks and Temples fusion (`--dataset TaT_a` / `TaT_i`)

- Both variants ignore `fusion.*` and `weakfilter.*` (they keep upstream's hard-coded thresholds and call
  `WeakVisFilter` with its defaults). APD now warns for every such option set to a non-default value.
- The per-source `diff` entries are created once per reference view and not reset per pixel, so a source
  that does not project into the image (or hits a masked or empty pixel) keeps the values of an earlier
  pixel and can still count as agreeing. Upstream behaviour; not changed.

## fusion.occlusion_test (off by default)

- The new default (`occlusion_tol = 0`, fusion's own agreement test decides what is "in front") rejects more
  points on the mouse1 ctrl scene than the old absolute 1e-4 (46448 vs 29945 of about 660k tested);
  `occlusion_tol = 1e-4` on top gives 14588. Which setting gives the better cloud and mesh has not been
  evaluated end to end (v19deeplayer scoring) yet.
