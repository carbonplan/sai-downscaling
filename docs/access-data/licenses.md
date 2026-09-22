# Licenses and citation

If you publish work based on this dataset, cite both our downscaled output and the source
simulations we built it from. The output carries a single license, and the input simulations carry
several, so we list the 2 separately below. Using any of this data also means agreeing to the
[Terms of Data Access](terms-of-data-access.md).

## Output data

Our downscaled output is licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). That
covers both published products, the downscaled 0.25° data and the coarse bias-corrected data, and
the authoritative copy of the license ships beside the data as `output/LICENSE.txt` on Source
Cooperative.

## Input data

Input datasets are subject to their respective licenses, which are summarized in the table below.
The first 2 columns use the same names as the published stores, so you can map a row onto the group
you opened. If you have questions or concerns regarding licensing of input datasets, please reach
out to the appropriate license holder directly.

| GCM | Scenario | License | Attribution and citation |
| --- | --- | --- | --- |
| `CESM2-WACCM6` | `historical` | CC-BY-4.0 | Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 CMIP historical. Version 20191104. Earth System Grid Federation. <https://doi.org/10.22033/ESGF/CMIP6.10071> |
| `CESM2-WACCM6` | `ssp245` | CC-BY-4.0 | Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 ScenarioMIP ssp245. Version 20191104. Earth System Grid Federation. <https://doi.org/10.22033/ESGF/CMIP6.10101> |
| `CESM2-WACCM6` | `g6_1p5k` |  | These simulations were run and provided by Walker Lee using the Community Earth System Model (<https://doi.org/10.5065/D67H1H0V>), which is developed and maintained by the National Center for Atmospheric Research. See also: Lee, W. R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M.: G6-1.5K-SAI and G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol injection strategy in the Geoengineering Model Intercomparison Project, Atmos. Chem. Phys., 26, 7463–7483, <https://doi.org/10.5194/acp-26-7463-2026>, 2026. |
| `CESM2-WACCM6` | `g6_1p5k_end` | CC-BY-4.0 | Zarakas, Claire (2026). Climate model output from simulation of abrupt termination of stratospheric aerosol injection. Version v1, <https://doi.org/10.5281/zenodo.22713138> |
| `UKESM1-1-LL` | `historical` | CC-BY-4.0 | Mulcahy, Jane; Rumbold, Steve; Tang, Yongming; Walton, Jeremy; Hardacre, Catherine; Stringer, Marc; Hill, Richard; Kuhlbrodt, Till; Jones, Colin (2022). MOHC UKESM1.1-LL model output prepared for CMIP6 CMIP. Version 20240824. Earth System Grid Federation. <https://doi.org/10.22033/ESGF/CMIP6.16781> |
| `UKESM1-1-LL` | `ssp245` | OGLv3 | These simulations were run by Andy Jones in collaboration with Jim Haywood and Matthew Henry, and provided by Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre. |
| `UKESM1-1-LL` | `g6_1p5k` | OGLv3 | These simulations were run by Andy Jones in collaboration with Jim Haywood and Matthew Henry, and provided by Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre. See also: Lee, W. R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M.: G6-1.5K-SAI and G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol injection strategy in the Geoengineering Model Intercomparison Project, Atmos. Chem. Phys., 26, 7463–7483, <https://doi.org/10.5194/acp-26-7463-2026>, 2026. |

:::{admonition} Input licenses differ by scenario
:class: warning

The input data is not covered by a single license. The UKESM1-1-LL `ssp245` and `g6_1p5k`
simulations are released under OGLv3 while everything else is CC-BY-4.0, so check the row for the
scenario you are using rather than assuming one license covers the input as a whole.
:::

| License | Full text |
| --- | --- |
| CC-BY-4.0 | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) |
| OGLv3 | [UK Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |
