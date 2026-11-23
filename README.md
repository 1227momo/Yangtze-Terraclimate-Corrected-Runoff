# Bias-corrected TerraClimate runoff depth for the Yangtze River Basin

This repository provides monthly bias-corrected TerraClimate runoff depth for the Yangtze River Basin upstream of Datong from January 2007 to December 2024. It includes GeoTIFF rasters, accompanying CSV tables, and Python scripts for monthly runoff evaluation and bias correction.

## Download the data

- [GeoTIFF rasters (V1.0.0)](https://github.com/1227momo/Yangtze-Terraclimate-Corrected-Runoff/releases/tag/V1.0.0). The **Assets** section contains 18 GeoTIFF files, one for each year from 2007 to 2024. Click a `.tif` filename to download it.
- [Accompanying CSV tables](csv/). Browse and download the published result tables from the `csv/` directory.

## GeoTIFF format

Files are named `TerraClimate_QM_ro_YYYY.tif`, where `YYYY` is the four-digit year. Each file contains 12 bands ordered from January (band 1) to December (band 12). Each band gives the total runoff depth for that month in millimeters (mm). The grid spacing is 0.0416667° (1/24°), and the NoData value is `-9999`.

## Monthly Python workflow

The [Python scripts](code/monthly/) evaluate raw TerraClimate runoff, fit quantile mapping models by zone and calendar month, reconstruct monthly runoff at five mainstream control stations, and evaluate model transfer at six tributary stations.

See the [workflow guide](code/monthly/README.md) for the required input CSVs, catchment area settings, Python dependencies, and execution order. Running the scripts requires station observations and extracted TerraClimate time series. The published CSV tables contain workflow results. The GeoTIFF rasters were generated separately using scaling factors produced by the Python workflow.
