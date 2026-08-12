# MetaDataExtraction

Check `Data/all_papers.csv` for each paper id's actual paper url. Everything else keys off `paper_id`.


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