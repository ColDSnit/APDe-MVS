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
