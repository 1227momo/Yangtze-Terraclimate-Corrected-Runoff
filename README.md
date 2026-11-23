# Bias-corrected TerraClimate runoff depth for the Yangtze River Basin

This repository provides monthly bias-corrected TerraClimate runoff depth for the Yangtze River Basin upstream of Datong from January 2007 to December 2024.

## Download the data

- [GeoTIFF rasters (V1.0.0)](https://github.com/1227momo/Yangtze-Terraclimate-Corrected-Runoff/releases/tag/V1.0.0). The **Assets** section contains 18 files, one for each year from 2007 to 2024. Click a `.tif` filename to download it.
- [Accompanying CSV tables](csv/). Open the `csv/` folder to browse and download the tables.

## GeoTIFF format

Files are named `TerraClimate_QM_ro_YYYY.tif`, where `YYYY` is the four-digit year. Each file contains 12 bands ordered from January (band 1) to December (band 12). Each band gives the total runoff depth for that month in millimeters (mm). The grid spacing is 0.0416667° (1/24°), and the NoData value is `-9999`.
