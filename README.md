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

Our line detection is built on **HAWP** — the `extract_lines` step uses the
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

The report is merged on write, so a partial re-run updates only the papers it
processed and leaves the other rows as they were. `--only id1,id2` and
`--only-file ids.txt` select papers explicitly.

