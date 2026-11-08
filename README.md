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

```bash
python -B -m unittest discover -s tests -v
```

## Third-party libraries and licences

Python 3.12/3.13. Versions are the ones the code was last tested with (Nov 2026);
licences are from each package's own metadata. Standard-library modules are not listed.

| Library | Version tested | Licence | Used for | Where |
|---|---|---|---|---|
| [NumPy](https://numpy.org) | 2.5.4 | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) | Array maths for images, boundaries, masks and statistics | everywhere |
| [Pillow](https://python-pillow.org) | 13.0.0 | MIT-CMU (HPND) | Opening TIFFs (including 1-bit CCITT scans), cropping, saving PNGs | `info extraction/`, `work/`, GeoLens |
| [PyTorch](https://pytorch.org) (`torch`) | 2.11–2.15 | BSD-3-Clause (wheel also bundles Apache-2.0, MIT, Zlib, BSL-1.0 and ISC parts) | Runs the YOLO swatch detector and trains/runs the U-Net region finder | `info extraction/`, `work/` |
| [torchvision](https://pytorch.org/vision) | 0.29.2 | BSD-3-Clause | Required by Ultralytics | indirect |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 8.4.207 | **AGPL-3.0** | YOLO detector for legend swatches and names (`models/swatch_detection.pt`); YOLO training notebook | `info extraction/interval_lithology.py`, `Model training/train.ipynb` |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr`) | 3.10.0 | Apache-2.0 | OCR of legend names and depth labels (ONNX models ship in the wheel) | `info extraction/legend_utils.py`, `depth_distance.py` |
| [ONNX Runtime](https://onnxruntime.ai) | 1.31.1 | MIT | Inference engine for RapidOCR | indirect |
| [OpenCV](https://opencv.org) (`opencv-python` / `-headless`) | 5.0.0.93 | Apache-2.0 | Resizing pages, connected components, drawing boxes | `work/`, `info extraction/unet_pipeline.py` |
| [tifffile](https://github.com/cgohlke/tifffile) | 2026.10.15 | BSD-3-Clause | Reading very tall LZW/palette TIFF pages for the U-Net | `work/pages.py` |
| [imagecodecs](https://github.com/cgohlke/imagecodecs) | 2026.10.10 | BSD-3-Clause | LZW decoding for tifffile | indirect (`work/`) |
| [Streamlit](https://streamlit.io) | 1.67.0 | Apache-2.0 | GeoLens web UI | `Viridien-GeoLens/` |
| [pandas](https://pandas.pydata.org) | 3.1.1 | BSD-3-Clause | Interval tables, statistics, CSV/Excel export | `Viridien-GeoLens/` |
| [Plotly](https://plotly.com/python) | 6.9.0 | MIT | Interactive profile and charts | `Viridien-GeoLens/` |
| [openpyxl](https://openpyxl.readthedocs.io) | 3.1.5 | MIT | Excel export (through pandas) | `Viridien-GeoLens/services/export_service.py` |
| [pytest](https://pytest.org) | 9.2.0 | MIT | Test runner (GeoLens and pipeline tests; `unittest` also works for `tests/`) | `tests/`, `Viridien-GeoLens/geolens/tests/` |
| Tkinter (Python standard library, Tcl/Tk) | bundled | PSF / Tcl-Tk BSD-style | Native "Browse…" file picker in GeoLens | `Viridien-GeoLens/services/local_picker.py` |
| [SciPy](https://scipy.org) | not in requirements | BSD-3-Clause | `find_peaks` in the standalone `edge_detection.py` script only | `info extraction/edge_detection.py` |
| `rapidocr_onnxruntime` | not in requirements | Apache-2.0 | Older OCR package, standalone `edge_detection.py` script only | `info extraction/edge_detection.py` |

**Models and data**

- `models/swatch_detection.pt`: a YOLO detector fine-tuned with Ultralytics from its pretrained
  YOLO26s weights. Ultralytics models and weights are AGPL-3.0 unless an Ultralytics Enterprise
  licence is obtained.
- `work/export/boxfinder_v0.2*.pt`: our own U-Net, trained from scratch on the Viridien dataset
  (no pretrained weights). See [work/README.md](work/README.md).
- The mud-log images and bounding boxes in `Datasets/` are provided by Viridien for the challenge
  and are not redistributed in this repository.

**AGPL-3.0 note:** Ultralytics is the only copyleft dependency. Using it for this open hackathon
project is fine. If this code, or a service built on it, is distributed or offered to others over a
network, AGPL-3.0 requires making the complete corresponding source available under the same
licence (or buying an Ultralytics Enterprise licence). Everything else is permissive
(MIT / BSD / Apache-2.0).
