# Schematic-to-Netlist Extraction & Manual Annotation

The **`code/info-extraction-pipeline/`** directory contains the pipeline for schematic-to-netlist extraction, as well as the GUI tool for manual annotation.

There are two main entry points in this directory:

- **`run_pipeline.py`** — the automatic extraction pipeline.
- **`manual_annotation_pipeline.py`** — the GUI for inspecting the output and
  correcting whatever the pipeline got wrong.

---

## Setup

Python 3.10. A CUDA GPU is recommended; CPU and Apple MPS also work.

```bash
conda env create -n <your-env-name> -f environment.yml
conda activate <your-env-name>
```
<!-- ### Pretrained weights

Not included in this repository. Download and unpack them into
`pretrainedWeights/` so the tree looks like this:

```
pretrainedWeights/
├── componentDetection/best.pt         # YOLO, 18 component classes
├── junctionJumpDetection/best.pt      # YOLO, junction/jump detection
├── junctionJumpDetection/best2cls.pt  # optional 2-class model, unioned in
├── orientationDetection/best_model.pt # ResNet-18 orientation classifier
└── hawp/                              # HAWPv3, including bestv3.pth
``` -->

<!-- TODO before release: weights download URL. -->

---

## Running the pipeline

Both directories are set in `pipeline_config.yaml`:

```yaml
paths:
  input_dir: images     # where your schematics are
  output_dir: output    # where everything is written
```

Then:

```bash
python run_pipeline.py                           # every image in input_dir
python run_pipeline.py --image path/to/a0.jpg    # one image
python run_pipeline.py --input-dir path/to/imgs  # a directory, searched recursively
```
Netlists end up in `<output_dir>/netlist/*.cir`.

The run has seven steps:

| Step | Does |
|------|------|
| `detect_components` | YOLO finds components and masks them out |
| `detect_orientation` | ResNet-18 labels each component `u`/`r`/`d`/`l` |
| `extract_lines` | Removes text, then extracts wires with [HAWPv3](https://github.com/cherubicXN/hawp) |
| `generate_nodes` | Groups wires into electrical nodes across junctions and jumps |
| `export_touches` | Records where each node touches which component |
| `build_incidence` | Assigns a net to every component pin |
| `build_netlist` | Writes the `.cir` |

Every step writes its artifacts and a visualization to `<output_dir>/`, so you
can re-run any part of the pipeline without redoing the rest:

```bash
python run_pipeline.py --step build_incidence --stem a0     # just one step
python run_pipeline.py --from-step generate_nodes --stem a0 # from there on
```

| Flag | Meaning |
|------|---------|
| `--config PATH` | Config file (default `pipeline_config.yaml`) |
| `--image PATH` | Process a single image |
| `--input-dir DIR` | Process every image in a directory |
| `--step STEP` | Run one step only, or `all` (default) |
| `--from-step STEP` | Run from this step through the end |
| `--stem NAME` | Restrict to this image stem; repeatable |
| `--device DEV` | Override the config device (`cpu`, `cuda`, `cuda:0`, `mps`) |
| `--force-ocr` | Recompute OCR text boxes instead of reusing the cache |
| `--no-copy-inputs` | Require inputs to already live in `<output_dir>/images` |

### When a netlist is not produced

If the pipeline cannot resolve a pin, it marks that component with a
**red flag** and refuses to emit a netlist for the image, reporting it as
skipped. Fix those images with the GUI below.

---

## Manual correction

```bash
python manual_annotation_pipeline.py
```

The left panel lists images that came out with red flags. Two ways to fix one:

- **Re-tune and re-run.** Adjust parameters in the right-hand panel, then
  "Re-run from \<step\>" reruns `run_pipeline.py` for just the selected images.
- **Edit directly.** Four editors — component bboxes, orientations, nodes
  (including hand-placed junction/jump boxes), and touches — write back to the
  same intermediate files the pipeline reads, so a later re-run picks up your
  corrections.

Parameter edits go to `.gui_session_config.yaml`; `pipeline_config.yaml` is
only modified by "Save as defaults".

> The GUI reads from `output/` as a hardcoded path. If you change `output_dir`
> in the config, the GUI will not find the results.

---

## Configuration

Besides the paths above, `pipeline_config.yaml` holds the model locations and
every tuning threshold, grouped by the step it affects and commented
individually. The ones worth reaching for first are `component_conf` (detection
sensitivity), `combined_hawp_threshold` and `min_line_length` (wire
extraction), and `node_union_dist` (how large a gap still counts as one node).

---

<!-- ## Repository layout

```
run_pipeline.py                     # automatic pipeline
manual_annotation_pipeline.py       # correction GUI
pipeline_common.py                  # config loading, paths, shared constants

step1_component_line_detection.py   # detect_components, extract_lines, generate_nodes
step2_component_orientation.py      # detect_orientation
step3_node_component_touches.py     # export_touches
step4_incidence_matrix.py           # build_incidence
step5_netlist.py                    # build_netlist

helper/                             # visualization used by the steps above
pipeline_config.yaml
environment.yml
```

The step modules are loaded by path so `run_pipeline.py` can inject the
configured directories before calling them; run them through `run_pipeline.py`
rather than directly.

--- -->

## Acknowledgements

Our line detection is built on HAWP — the `extract_lines` step uses the
HAWPv3 wireframe parser and the authors' released checkpoint. We are grateful
to Nan Xue and colleagues for making their code and models publicly available.

- Code: <https://github.com/cherubicXN/hawp>
- Paper: <https://arxiv.org/abs/2210.12971>

We also thank the authors of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
and [EasyOCR](https://github.com/JaidedAI/EasyOCR), which the detection and
text-removal steps rely on.

---
<!-- 
## Citation



If you use this pipeline, please also cite HAWP:

```bibtex
@article{HAWP-journal,
  title   = {Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning},
  author  = {Nan Xue and Tianfu Wu and Song Bai and Fu-Dong Wang and Gui-Song Xia and Liangpei Zhang and Philip H.S. Torr},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
  year    = {2023}
}

@inproceedings{HAWP,
  title     = {Holistically-Attracted Wireframe Parsing},
  author    = {Nan Xue and Tianfu Wu and Song Bai and Fu-Dong Wang and Gui-Song Xia and Liangpei Zhang and Philip H.S. Torr},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

## License -->

# MetaDataExtraction

The **`code/MetaDataExtration/`** directory contains the metadata extraction pipeline and its output results.

In this directory, check `Data/all_papers.csv` for each paper id's actual paper url. Everything else keys off `paper_id`.


## Output

Results are written to `<output_dir>/<paper_id>/`, one subfolder per paper:

- `analysis.json` — the full result (machine/downstream contract).
- `summary.txt` / `summary.json` — the concise, human-readable view.

## Running the extraction

Single paper:

```
python cli.py analyze <path/to/your/paper.pdf>
```

Every PDF in a folder (figure cropping is skipped by default — text, captions and
circuit context only; add `--save-figures` to also crop the figure images):

```
python run_batch.py <path/to/papers> --out <output_dir>
```

A per-paper failure is caught and recorded rather than aborting the run. Two
report files land in `<output_dir>/_report/`:

- `extraction_report.csv` — one row per paper: `status` (`ok` / `incomplete` /
  `failed`), the `issues` tags, figure/section counts and the extracted title.
- `failures.json` — hard failures with tracebacks, plus the incomplete list.

`status` is `incomplete` when a critical field came out empty (no title,
sections, figures, circuit type or recommendation); `issues` also records
non-critical gaps (`no_authors`, `no_abstract`, `no_key_specs`).

To re-run only the papers that had problems, without touching results that
already came out clean:

```
python run_batch.py <path/to/papers> --out <output_dir> --rerun-failed-from <output_dir>/_report/extraction_report.csv
```
# LLM-repair

Solution-guided LLM repair loop for schematic-to-netlist extraction.

Given (a) a machine-generated SPICE netlist and (b) the schematic image it came
from, this pipeline validates the netlist with a three-level checker
(syntax → connectivity → ngspice operating-point simulation) and, when the
netlist fails, asks a panel of vision LLMs to propose corrected netlists. A
candidate is promoted to *solution* by topology consensus across models, and is
only accepted as final if it then passes Level 1–3 itself. Optionally, a
rename-only LLM pass gives nets and instances semantic names taken from the
schematic without changing topology.

Nothing here is dataset-specific: all input and output locations are command
line arguments.

---

## 1. Requirements

**Python** 3.10 or newer (developed and tested on 3.14).

**Python packages**

```bash
pip install networkx pillow certifi
```

`certifi` is optional — it is imported inside a `try` block and only used to
locate a CA bundle.

**System tools**

| Tool | Required for | Notes |
|---|---|---|
| `ngspice` | Level 3 simulation | Hard requirement; the validator shells out to `ngspice -b` |
| `curl` | all LLM calls | LLM requests are issued via `curl` subprocesses, not `requests` |
| `tesseract` | `verify_schematic_netlist.py` OCR only | Not needed for the repair loop |

**API keys** (read from the environment; there are no defaults)

```bash
export OPENAI_API_KEY=sk-...      # gpt-4.1-nano, gpt-4.1-mini
export GOOGLE_API_KEY=...         # gemini-3.1-flash-lite, gemini-3.1-pro-preview
```

Without a key the LLM candidates fail individually (shown as `E` in the progress
table) and every case ends `unresolved_after_max_iterations`. The local
validation path needs no key at all — see `--openrouter-max-cases 0` below.

---

## 2. Files

| File | Role |
|---|---|
| `run_solution_guided_llm_repair.py` | **Main entry point.** The consensus-based repair loop. |
| `run_iterative_llm_repair.py` | Earlier single-model iterative loop (OpenRouter). Also provides `validate_netlist_file()` used by the main script. |
| `run_general_validation.py` | Batch Level 1–3 validation over a directory of `.cir` files, no LLM involved. |
| `validate.py` | The validation engine itself: netlist parsing, port inference, SPICE deck generation, ngspice invocation, OP/DC/AC/TRAN result parsing. |
| `netlist_parser.py` | Shared netlist tokenizer / component model / graph construction. |
| `compare_netlist_rule_levels.py` | Rule 1–5 comparison of generated netlists against ground truth. |
| `verify_schematic_netlist.py` | Standalone OCR + LLM schematic verifier; also hosts `openrouter_chat()`, `resolve_case_image()`, `parse_id_ranges()`. |
| `generate_visualized_indexed.py` | Utility: draws bounding-box annotations on schematic images (produces the optional `--visualized-dir` input). |

Import graph:

```
netlist_parser.py
  ├── compare_netlist_rule_levels.py
  ├── verify_schematic_netlist.py
  └── run_general_validation.py ── validate.py
        └── run_iterative_llm_repair.py
              └── run_solution_guided_llm_repair.py
```

---

## 3. Inputs

The repair loop needs two directories, both keyed by an integer case ID:

* `--generated-dir` — the netlists to repair. Files are `<id>.cir`, either
  zero-padded to six digits (`000001.cir`) or plain (`1.cir`). Both are tried.
* `--image-dir` — the schematic images. Files are `<id>.png`, `<id>.jpg` or
  `<id>.jpeg`, same ID conventions.

Optional:

* `--ground-truth-dir` — reference netlists. Only used to *report* Rule 1–5
  scores; ground truth is never shown to the LLM and never affects the repair
  decision (`rules_affect_logic: false` in the output config).
* `--visualized-dir` — bbox-annotated copies of the images, passed to the LLM as
  a second image. Produce them with `generate_visualized_indexed.py`. Omit to
  disable.
* `--dataset-root` — fallback image lookup used when `--image-dir` does not
  contain the case (supports `<root>/<id>/Book<id>.png` and
  `RelativePath<id>.txt` layouts).
* `--id-prefix` — filename prefix, e.g. `a` for `a200.cir`.

Netlist line format (the same format the LLM is asked to emit):

```
M<name> <drain> <gate> <source> <bulk> <pmos|nmos> [W] [L]
Q<name> <collector> <base> <emitter> <pnp|npn> [value]
R/C/L/I/V<name> <node1> <node2> [value]
```

---

## 4. Quick start

Reproduce a run like the `LLM_out/` directory of the RNDatasetIndexed dataset:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...

python3 run_solution_guided_llm_repair.py 1101-1300 \
    --generated-dir /path/to/RNDatasetIndexed/netlist \
    --image-dir     /path/to/RNDatasetIndexed/images \
    --dataset-root  /path/to/RNDatasetIndexed/images \
    --out-dir       /path/to/RNDatasetIndexed/LLM_out \
    --max-iterations 3 \
    --timeout-s 600 \
    --openrouter-max-cases 700 \
    --min-solution-consensus 2 \
    --rename-on-pass
```

Positional IDs accept single values and inclusive ranges (`1 5 1101-1300`).
Omit them entirely to run every `*.cir` found in `--generated-dir`.

**Dry run with no API cost.** `--openrouter-max-cases 0` disables the LLM for
every case, so only the local Level 1–3 validation runs. Useful to check that
ngspice and the imports are working:

```bash
python3 run_solution_guided_llm_repair.py 1 2 3 \
    --generated-dir /path/to/netlist \
    --image-dir     /path/to/images \
    --dataset-root  /path/to/images \
    --out-dir /tmp/test_out \
    --openrouter-max-cases 0
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--max-iterations` | 3 | Repair rounds per case |
| `--min-solution-consensus` | 2 | How many models must agree on a topology before it is treated as the solution |
| `--openrouter-max-cases` | 100 | Budget guard: only the first *N* cases may call the LLM; the rest are validated locally |
| `--timeout-s` | 600 | Per-request timeout passed to `curl` |
| `--rename-on-pass` | off | After a candidate passes Level 1–3, run a rename-only LLM pass for semantic net/instance names |
| `--rule1-mode` | `none` | `none`: ignore Rule 1 when selecting. `skip`: if the original already passes Rule 1, skip LLM iterations entirely. `reject`: discard candidates that break a Rule 1 the original satisfied |
| `--ignore-types` | — | Comma-separated component types excluded from Rule 1–5 scoring, e.g. `current_source` |

### Models

Defined in `CANDIDATE_MODELS` near the top of
`run_solution_guided_llm_repair.py`:

```python
("gpt-4.1-nano",           "openai")
("gpt-4.1-mini",           "openai")
("gemini-3.1-flash-lite",  "gemini")
("gemini-3.1-pro-preview", "gemini")   # JUDGE_MODEL
```

The first three are the *fast* panel. If they reach no consensus, the judge
model is called as a fallback. Backends map to
`https://api.openai.com/v1/chat/completions` and
`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
(the OpenAI-compatible Gemini endpoint).

---

## 5. Output layout

Everything is written under `--out-dir`, one directory per case ID
(zero-padded to six digits) plus a top-level `summary.json`.

```
<out-dir>/
├── summary.json                      # config + aggregate summary + all case records
└── 000160/
    ├── result.json                   # full record for this case
    ├── original_evaluation/          # Level 1-3 run on the input netlist
    │   ├── validation_cases/000160/000160.cir
    │   └── simulation/tc000160/      # generated SPICE decks + ngspice logs
    ├── iter_01/
    │   ├── current_original.cir      # netlist fed to the LLM this round
    │   ├── current_validation/
    │   ├── llm_raw/
    │   │   ├── prompt.txt
    │   │   └── candidate_0N_<model>.raw.txt
    │   ├── candidate_0N.cir          # parsed candidate per model
    │   ├── candidate_0N_validation/
    │   ├── renamed_candidate.cir     # only with --rename-on-pass
    │   └── rename_raw/
    ├── iter_02/, iter_03/            # same shape; `current_no_solution_iter_N.cir`
    ├── final_validation/
    └── final_unresolved.cir          # or the accepted final netlist
```

`result.json` per case contains `generated` (input + its validation),
`image_path` / `image_source`, `ground_truth`, `iterations[]` (every candidate,
its raw LLM text, parsed netlist and validation), and `final`.

`final.status` is one of:

| Status | Meaning |
|---|---|
| `original_passed_level_1_3` | Input netlist was already valid; no LLM needed |
| `final_solution_level_1_3_passed` | An LLM candidate was accepted |
| `unresolved_after_max_iterations` | No candidate passed within the iteration budget |
| `skipped_original_passes_rule1` | `--rule1-mode skip` preserved the original topology |

With `--rename-on-pass`, a suffix is appended: `_renamed` (the renamed netlist
still passes Level 1–3), `_rename_failed`, or `_rename_attempted`.

### Progress table

While running, one row is printed per candidate:

```
   #      ID  it  model                      L1   L2   L3    DC    AC    TR   R1   R2   R3   R4   R5
   1    1101   0  [generated]                 Y    n    Y                      n    n    n    n    n
   1    1101   1  gpt-4.1-nano                Y    n    n                      n    n    n    n    n
   1    1101      => unresolved_after_max_iterations  ...
```

`Y` pass, `n` fail, `E` the model call itself errored.

---

## 6. Validation levels and rules

**Level 1–3** (`validate.py`, always run locally):

| Level | Check |
|---|---|
| L1 | Syntax — every line parses into a known device with the right arity |
| L2 | Connectivity — a ground reference (`VSS`/`0`) and `VDD` exist, no internal node has only one connection, no duplicate instance names, every inferred port actually appears |
| L3 | Simulation — an ideal level-1 CMOS deck converges in ngspice and the operating point is physically valid |

`DC` / `AC` / `TR` are additional functional analyses, reported but not required
for acceptance. In the repair loop (`validate_netlist_file()`), Level 3 sweeps
the bias current over `_ISOURCE_SWEEP_BASES = [1, 2, 5, 10, 20, 50] µA` and keeps
the first magnitude that yields a valid operating point; `run_general_validation.py`
uses the single 1 µA default instead, so its pass rate can be slightly lower.

**Rules 1–5** (`compare_netlist_rule_levels.py`, only when
`--ground-truth-dir` is given) score a netlist against ground truth with
increasing strictness:

| Rule | Requires |
|---|---|
| `rule1_topology_only` | Graph isomorphism; all net names and instance indexes ignored |
| `rule1_2_supply_nets` | Topology + VDD/GND/VSS-like labels |
| `rule1_3_special_nets` | Topology + all port-like labels (VIN, VOUT, bias, clock, …) |
| `rule1_4_transistor_indexes` | Rule 3 + MOSFET/BJT instance indexes |
| `rule1_5_all_component_indexes` | Rule 4 + every component instance index |

These are reporting-only; they never influence which candidate is chosen
(except under `--rule1-mode skip|reject`).

---

## 7. Other entry points

**Batch validation, no LLM:**

```bash
python3 run_general_validation.py \
    --dataset-dir /path/to/netlist \
    --workers 8 --limit 100
```

Writes `simulation_general/general_validation_<timestamp>.json` next to the
script and prints a pass-rate table.

**Rule 1–5 comparison against ground truth:**

```bash
python3 compare_netlist_rule_levels.py \
    --generated-dir    /path/to/netlist_generated \
    --ground-truth-dir /path/to/netlist_ground_truth \
    --json-out         /path/to/rule_level_comparison.json
```

**Standalone OCR + LLM verifier** (needs `tesseract` and
`OPENROUTER_API_KEY`):

```bash
python3 verify_schematic_netlist.py 1-50 \
    --dataset-root /path/to/dataset \
    --netlist-dir  /path/to/netlist \
    --json-out     /path/to/verify_report.json
```

**Bbox-annotated images** for `--visualized-dir`:

```bash
python3 generate_visualized_indexed.py --base-dir /path/to/dataset
```


The report is merged on write, so a partial re-run updates only the papers it
processed and leaves the other rows as they were. `--only id1,id2` and
`--only-file ids.txt` select papers explicitly.

