'''
Purpose: Write docs/RUNTIME_OPTIONS.md from the option schema that the APD binary prints with
         --dump_options_json, so the reference table can never drift from the code.
Status:  Working
Future:  None
Usage:   python tools/gen_options_doc.py <path to APD binary>
'''
import json                                    # parses the schema printed by APD
import subprocess                              # runs the APD binary
import sys                                     # command-line arguments
from pathlib import Path                       # cross-platform paths

INTRO = """# APD run-time options

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

## Per-pixel depth prior (`prior.*`)

With `prior.enable = true`, APD reads `<dense>/depth_prior/<ref>.bin` for each reference view:

- Format: APD BinMat, i.e. four int32 values (version 1, rows, cols, OpenCV type) followed by the row-major
  pixel data. Type `CV_32FC1` holds a prior depth, turned into the band
  `[depth - prior.band_near, depth + prior.band_far]`; type `CV_32FC2` holds the band itself as `(lo, hi)`,
  and the two band options are then unused. Any other type disables the prior for that view, with a warning.
- Values are camera-frame z-depths (the quantity in `depths.bin`, not the distance along the ray) in the
  world units of the cam files. They are absolute: `depth.range_scale_*` and `depth.range_pad` do not apply.
- Size and pixel convention: the map must have the size of `<dense>/images/<ref>` (the undistorted scene
  image, before APD's own scaling); any other size disables the prior for that view, with a warning. Pixel
  (x, y) is the ray through image position (x, y) of the cam-file K (pixel centres at integer coordinates).
- 0, a non-finite value, a non-positive `lo` or `hi <= lo` mean "no prior here": that pixel keeps the view's
  depth range. A file with no valid pixel, a missing file or an unreadable file disables the prior for the
  view with a warning.
- At each scale the map is pooled to the working size: a working pixel takes the minimum `lo` and maximum
  `hi` of the full-size pixels whose centres lie in its footprint (at full size, its own band).
- Inside the band APD draws its random initial depths, refinement candidates and perturbations, accepts
  propagated planes, evaluates the weak/strong cost curve, runs the final local refinement, and finally
  sets depths outside the band to 0. Side effects of a narrow band: the weak/strong cost curve sees only the
  depths inside the band, so more pixels are classified STRONG; the median filter of STRONG pixels is not
  confined to the band, so a filtered depth that leaves it becomes 0; when the inherited depth of a later
  round lies outside the band (or is 0), the pixel is re-seeded randomly inside its band, and with a prior
  loaded a pixel without a band is re-seeded inside the view range in the same way.
- Units: the `prior.band_*` defaults (0.0005) are 0.5 mm only when the world unit is the metre.
- With a prior loaded, the depth perturbation is always clipped to the pixel's band (`pm.perturbation_clip`).

## Reference

| Option | Type | Default | Range / choices | Unit | Stage | Description |
|---|---|---|---|---|---|---|
"""


def main() -> None:
    exe = Path(sys.argv[1])                                                   # APD binary to query
    raw = subprocess.run([str(exe), "--dump_options_json"], capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout                 # schema as JSON text
    rows = []
    for opt in json.loads(raw)["options"]:
        if "choices" in opt:
            limits = ", ".join(opt["choices"])                                # enum: list the names
        elif "min" in opt:
            limits = f"{opt['min']:g} .. {opt['max']:g}"                      # numeric: inclusive range
        else:
            limits = ""
        default = str(opt["default"]).lower() if isinstance(opt["default"], bool) else str(opt["default"])
        rows.append(f"| `{opt['name']}` | {opt['type']} | {default} | {limits} | {opt['unit']} | "
                    f"{opt['stage']} | {opt['help']} |")
    out = Path(__file__).resolve().parent.parent / "docs" / "RUNTIME_OPTIONS.md"
    out.parent.mkdir(exist_ok=True)                                           # docs/ may not exist yet
    out.write_text(INTRO + "\n".join(rows) + "\n", encoding="utf-8")          # explicit UTF-8, LF line ends
    print(f"wrote {out} ({len(rows)} options)")


if __name__ == "__main__":
    main()
