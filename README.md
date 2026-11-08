# StackUnderflow_Submission
Stack Underflows teams submission for HackTheBurgh 2026

## Complete lithology pipeline

Input: **one full-document TIFF and its three-region bounding-box JSON**.
The pipeline crops the regions, detects RGB/texture intervals, groups returning
patterns, matches the document's legend, and calibrates interval depths.

```bash
python 'info extraction/pipeline.py' \
  'Dataset/dataset/images/Rio_Lavaca_1748_Unit_1_241_MUD.tif' \
  'Dataset/dataset/bbox_annotation/Rio_Lavaca_1748_Unit_1_241_MUD.json'
```

Results are written to `crops/pipeline_output/<TIFF stem>/`:
`intervals.csv`, `intervals.json`, an annotated `intervals.png`, calibration and
legend review files, native crops, and a `run.json` provenance/status report.
Final pixel boundaries default to **original-document coordinates**; crop-local
coordinates are retained in extra columns. Use `--coordinate-system crop` for
the earlier crop-local convention. Use `--overwrite` to archive an existing run
and start again without deleting previous results.

- [Complete pipeline usage and output format](info%20extraction/pipeline.md)
- [RGB interval boundary detection](info%20extraction/texture_boundaries.md)
- [Interval clustering and lithology identification](info%20extraction/interval_lithology.md)
- [Depth calibration and cropped-input export](info%20extraction/final_output.md)

Superseded utilities were preserved in [`wastebin/`](wastebin/README.md).
The active pipeline follows `Flow Diagram.drawio`.

```bash
python -B -m unittest discover -s tests -v
```
