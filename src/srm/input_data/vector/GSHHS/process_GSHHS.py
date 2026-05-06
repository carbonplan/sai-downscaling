import subprocess
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd

# grab this high-res coastline dataset from NOAA: https://www.ngdc.noaa.gov/mgg/shorelines/shorelines.html
URL = "https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip"
S3_DEST = "s3://carbonplan-srm/input/vector/GSHHS/GSHHS.parquet/"

zip_path = Path("gshhg-shp.zip")
extract_dir = Path("gshhg-shp")
out_path = Path("GSHHS.parquet")

if __name__ == "__main__":
    urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    shp = next(extract_dir.rglob("GSHHS_f_L1.shp"))
    gdf = gpd.read_file(shp)[["geometry"]].rename_geometry("geom")
    gdf.to_parquet(out_path)

    subprocess.run(["aws", "s3", "cp", str(out_path), S3_DEST], check=True)
