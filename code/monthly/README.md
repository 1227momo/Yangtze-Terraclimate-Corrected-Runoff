# Monthly TerraClimate runoff evaluation and bias correction

These five Python scripts evaluate monthly TerraClimate runoff depth, fit quantile mapping (QM) by independent zone and calendar month, reconstruct runoff depth at five mainstream control stations, and evaluate how the fitted models transfer to tributary stations in the Yangtze River Basin upstream of Datong.

## Setup

Use Python 3.11 and install the tested dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

The scripts locate `inputs/` and `outputs/` next to their own files. Place the CSVs in the folders below or change the path variables at the top of the scripts. The scripts create their output folders automatically. Use UTF-8 or UTF-8 with BOM when preparing CSVs. Input column names and station IDs are case-sensitive in some scripts.

Replace each `None` in `areas.py` with the actual contributing area in km². Enter the five mainstream catchment areas and the five independent zone areas. Steps 01 and 03 use the same zone area entries, and step 03 uses the mainstream catchment areas for reconstruction.

## Input CSVs

| Folder under `inputs/` | Required data and columns |
| --- | --- |
| `terraclimate/` | `TC_monthly_ro_1958_2024.csv` with `year`, `month`, and one column per station ID used by steps 00 and 04. Mainstream IDs are `pingshan`, `cuntan`, `yichang`, `hankou`, and `datong`. Step 04 lists the expected tributary IDs in `TC_STATION_COLUMNS`. Runoff depth is in mm. |
| `terraclimate/` | `TC_monthly_ro_Z1_Z5_1958_2024.csv` with `year`, `month`, and `Z1` through `Z5` in mm. |
| `observed_runoff/mainstream/` | Monthly observed runoff depth in mm. Pingshan, Yichang, and Hankou use `date` and `obs_ro_mm`. Cuntan uses `year`, `month`, and `cuntan_ro_mm`. Datong uses `data` and `obs_ro_monthly`. Step 03 can also read Datong monthly volume and convert it using the area in `areas.py`. See `OBS_PATHS` in steps 00 and 03 for the station-specific filenames. The two steps use different source filenames for Hankou and Datong. |
| `observed_volume/mainstream/` | Five station CSVs named in `OBS_FLOW_PATHS` in step 01. Each CSV contains `year`, `month`, and a monthly volume column such as `pingshan_volume_1e8m3`. Volume is in units of 10⁸ m³. |
| `observed_runoff/tributaries/` | One CSV per station listed in `TRIBUTARY_OBS_FILES` in step 04. Use `year`, `month`, and `obs_ro_mm` for runoff depth in mm. The script requires all configured observation tables by default. |

This bundle contains the scripts and documentation. The station observations and extracted TerraClimate input CSVs are required to reproduce the computations. The corrected CSV tables in the repository's `csv/` directory are result tables.

## Run the scripts

From the directory containing this README:

```bash
python -X utf8 00_evaluate_raw_mainstream.py
python -X utf8 01_evaluate_raw_zones.py
python -X utf8 02_fit_zone_month_qm.py
python -X utf8 03_reconstruct_mainstream.py
python -X utf8 04_validate_tributaries.py
```

| Step | Operation | Output folder under `outputs/` |
| --- | --- | --- |
| 00 | Compare raw TerraClimate runoff depth with observations at the five mainstream stations over the available overlapping months. This diagnostic can run independently. | `00_raw_mainstream/` |
| 01 | Derive zone-level monthly runoff depth from differences in observed mainstream water volumes. Exclude negative differences for the affected zone and month, then compare with raw TerraClimate. | `01_raw_zones/` |
| 02 | Read `OBS_monthly_volume_ro_Z1_Z5_wide.csv` from step 01. Train on valid overlapping months in 2007–2024. Use leave-one-year-out cross-validation to select a model for each zone and calendar month, refit it on all eligible samples, and apply it to the full series provided in the TerraClimate input. | `02_zone_month_qm/` |
| 03 | Read the corrected zone table from step 02 and reconstruct raw and corrected runoff depth at the five mainstream stations by area-weighted aggregation. Evaluate the results over 2007–2024. | `03_mainstream_reconstruction/` |
| 04 | Read the final model summaries and dense mapping curves from step 02. Apply the fitted zone and calendar-month models to the corresponding tributary catchments. Evaluate the Raw and QM series over the valid overlapping months for each station. | `04_tributary_validation/` |

Step 02 saves observation coverage, cross-validation results, and method selections in `00_model_selection/`. It saves the final models and curves in `01_final_models/`, corrected zone runoff and GEE scaling factors in `02_corrected_runoff/`, and zone evaluation metrics in `03_zone_validation/`. The Python workflow produces tabular outputs. GeoTIFF generation requires applying the scaling factors in a separate raster workflow.

## Model selection and tributary validation

Step 02 first shortlists candidate methods with finite NSE within 0.01 of the highest NSE for the zone and calendar month. It then selects the lowest RMSE among those candidates, using absolute PBIAS and then absolute bias to break ties. If all NSE values are unavailable, the script applies the same ordering to candidates with finite RMSE.

Step 04 uses six tributary gauges. The evaluation uses months with valid observations and both TerraClimate estimates.

| Independent zone | Tributary gauges |
| --- | --- |
| Z2 | Gaochang |
| Z4 | Chenglingji, Xiangtan, Taoyuan |
| Z5 | Hukou, Waizhou |
