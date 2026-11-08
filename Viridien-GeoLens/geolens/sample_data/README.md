# Sample data (illustrative only)

These CSV files are **synthetic** and exist purely to exercise GeoLens during
development and testing. They are not real geological data and must not be used
for any geological interpretation.

| File | Purpose |
| --- | --- |
| `well_a.csv` | Baseline document: Shale / Siltstone / Sandstone / Limestone |
| `well_b.csv` | Overlapping lithologies with a new type (`Dolomite`) and different cluster IDs |
| `well_c.csv` | Mostly different lithologies (`Evaporite`, `Conglomerate`) |
| `well_d_messy.csv` | Deliberate data-quality issues: missing lithology, overlapping intervals, gaps, duplicate-ish records |
| `company_example.csv` | The company's real export format, which uses `bottom_depth` instead of `base_depth` |

The `well_*.csv` files contain the six canonical columns:
`top_y`, `bottom_y`, `cluster_id`, `lithology`, `top_depth`, `base_depth`.

`company_example.csv` is a copy of the example file the company provided. It
uses `bottom_depth` in place of `base_depth`; GeoLens maps this automatically
(see "Alternative column names" in the main README).
