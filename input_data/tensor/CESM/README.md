# CESM: G6-1.5K and WACCM-SSP245


## Data transfer
NetCDF files for G6-1.5K and WACCM-SSP245 was moved from NCAR Derecho to s3 using rclone.
- s3://carbonplan-srm/input/tensor/CESM-G6-1.5K/*.nc
- s3://carbonplan-srm/input/tensor/CESM2-WACCM-SSP245/*.nc

## Virtualization and rechunked Icechunk store
In the scripts `create_virtual_and_icechukn_CESM_****_.py>`, a Virtual Zarr store of these NetCDF files were created using VirtualiZarr. This gives us a data cube of each dataset and avoids using `open_mfdataset`. The source NetCDF had very small chunking (~kb's) so the Virtual Zarr datacube was rechunked and written to Icechunk with ~100Mb chunks.

Those data can be found:

### Virtual
- `'s3://carbonplan-srm/input/tensor/CESM2-WACCM-SSP245/icechunk/virtual_icechunk'`
- `'s3://carbonplan-srm/input/tensor/CESM-G6-1.5K/icechunk/virtual_icechunk'`


### Icechunk
- `'s3://carbonplan-srm/input/tensor/CESM2-WACCM-SSP245/icechunk/icechunk'`
- `'s3://carbonplan-srm/input/tensor/CESM-G6-1.5K/icechunk/icechunk'`




