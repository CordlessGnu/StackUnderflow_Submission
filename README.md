# StackUnderflow_Submission
Stack Underflows teams submission for HackTheBurgh 2026

## GeoLens local demo

```bash
python -m pip install -r requirements.txt
python run_geolens.py
```

Open **http://127.0.0.1:8501**. In **Single Analysis**, keep the existing CSV mode
or select **TIFF + JSON**, choose/upload the two files, select the output directory,
and click **Run extraction**. Progress runs in the background; on success GeoLens
loads the generated CSV into its existing analytics. Native Browse dialogs are
optional—local paths can always be entered manually. Each run gets a fresh output
subfolder; previous runs are not overwritten.

See [GeoLens usage](Viridien-GeoLens/README.md) for settings, cancellation, download
and collection workflows. Depth rounding follows the pipeline default (10 units);
choose **0** in Extraction settings to preserve thin-interval precision.

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
- [Hybrid grouping and legend matching](info%20extraction/interval_lithology.md)
- [Depth calibration and cropped-input export](info%20extraction/final_output.md)

Superseded utilities were preserved in [`wastebin/`](wastebin/README.md).
The active pipeline follows `Flow Diagram.drawio`. Grouping/naming now defaults
to cached descriptor shortlists plus multiscale seam verification. Use
`--matcher descriptor` for the earlier comparison method. Nonblank unknown labels
use the nonempty runner-up, marked `lithology_fallback=True`; original match
statuses remain available. Rows with `match_status=blank` keep an empty
`lithology` cell and never receive a runner-up label.

```bash
python -B -m unittest discover -s tests -v
```
