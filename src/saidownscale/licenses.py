"""Canonical license and attribution for the SAI Downscaling inputs and output.

This module is the single source of truth. The store attrs written by
``scripts/add_store_metadata.py`` and the table in ``docs/access-data/licenses.md`` both derive
from it, and ``tests/test_terms_of_data_access.py`` fails if the two disagree. Edit here first,
then re-sync the docs table. The ``LICENSE.txt`` files published beside the data on Source
Cooperative are not generated from this module yet.

Licenses are named by their SPDX identifier so that generic tooling can resolve them. A license of
``None`` means we assert no license for that simulation, which is different from not knowing: the
attribution is still recorded, and nothing downstream may invent a license in its place.
"""

from __future__ import annotations

from dataclasses import dataclass

CC_BY_4_0 = "CC-BY-4.0"
OGL_UK_3_0 = "OGL-UK-3.0"

#: Markdown spelling in ``docs/access-data/licenses.md`` -> SPDX identifier.
SPDX_BY_DOCS_NAME: dict[str, str] = {"CC-BY-4.0": CC_BY_4_0, "OGLv3": OGL_UK_3_0}

LICENSE_URLS: dict[str, str] = {
    CC_BY_4_0: "https://creativecommons.org/licenses/by/4.0/",
    OGL_UK_3_0: "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
}

#: Our downscaled output carries one license, whatever it was derived from. The upstream
#: simulation still has to be credited, which is what ``references`` below is for.
OUTPUT_LICENSE = CC_BY_4_0

INSTITUTION = "CarbonPlan"


@dataclass(frozen=True)
class Attribution:
    """The license and citation for one input simulation.

    Parameters
    ----------
    license : str or None
        SPDX identifier, e.g. ``"CC-BY-4.0"``. ``None`` where we assert no license.
    references : str
        Citation text, written to the CF ``references`` attribute. Always present, because a
        simulation we publish is always one we can credit.
    """

    license: str | None
    references: str

    @property
    def license_url(self) -> str | None:
        """Return the canonical URL for :attr:`license`, or None when none is asserted."""
        return LICENSE_URLS[self.license] if self.license else None


#: Keyed on ``(gcm, scenario_group)``, using the spellings the stores themselves use.
INPUT_ATTRIBUTION: dict[tuple[str, str], Attribution] = {
    ("CESM2-WACCM6", "historical"): Attribution(
        license=CC_BY_4_0,
        references=(
            "Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 "
            "CMIP historical. Version 20191104. Earth System Grid Federation. "
            "https://doi.org/10.22033/ESGF/CMIP6.10071"
        ),
    ),
    ("CESM2-WACCM6", "ssp245"): Attribution(
        license=CC_BY_4_0,
        references=(
            "Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 "
            "ScenarioMIP ssp245. Version 20191104. Earth System Grid Federation. "
            "https://doi.org/10.22033/ESGF/CMIP6.10101"
        ),
    ),
    ("CESM2-WACCM6", "g6_1p5k"): Attribution(
        # No license is asserted for this simulation. See docs/access-data/licenses.md;
        # nothing downstream may substitute one.
        license=None,
        references=(
            "These simulations were run and provided by Walker Lee using the Community Earth "
            "System Model (https://doi.org/10.5065/D67H1H0V), which is developed and "
            "maintained by the National Center for Atmospheric Research. See also: Lee, W. "
            "R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., "
            "Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M.: G6-1.5K-SAI "
            "and G6sulfur: changes in impacts and uncertainty depending on stratospheric "
            "aerosol injection strategy in the Geoengineering Model Intercomparison Project, "
            "Atmos. Chem. Phys., 26, 7463–7483, https://doi.org/10.5194/acp-26-7463-2026, "
            "2026."
        ),
    ),
    ("CESM2-WACCM6", "g6_1p5k_end"): Attribution(
        license=CC_BY_4_0,
        references=(
            "Zarakas, Claire (2026). Climate model output from simulation of abrupt "
            "termination of stratospheric aerosol injection. Version v1, "
            "https://doi.org/10.5281/zenodo.22713138"
        ),
    ),
    ("UKESM1-1-LL", "historical"): Attribution(
        license=CC_BY_4_0,
        references=(
            "Mulcahy, Jane; Rumbold, Steve; Tang, Yongming; Walton, Jeremy; Hardacre, "
            "Catherine; Stringer, Marc; Hill, Richard; Kuhlbrodt, Till; Jones, Colin (2022). "
            "MOHC UKESM1.1-LL model output prepared for CMIP6 CMIP. Version 20240824. Earth "
            "System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.16781"
        ),
    ),
    ("UKESM1-1-LL", "ssp245"): Attribution(
        license=OGL_UK_3_0,
        references=(
            "These simulations were run by Andy Jones in collaboration with Jim Haywood and "
            "Matthew Henry, and provided by Matthew Henry. The UK Earth System Model is "
            "developed and maintained by the Met Office Hadley Centre."
        ),
    ),
    ("UKESM1-1-LL", "g6_1p5k"): Attribution(
        license=OGL_UK_3_0,
        references=(
            "These simulations were run by Andy Jones in collaboration with Jim Haywood and "
            "Matthew Henry, and provided by Matthew Henry. The UK Earth System Model is "
            "developed and maintained by the Met Office Hadley Centre. See also: Lee, W. R., "
            "Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, "
            "T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M.: G6-1.5K-SAI and "
            "G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol "
            "injection strategy in the Geoengineering Model Intercomparison Project, Atmos. "
            "Chem. Phys., 26, 7463–7483, https://doi.org/10.5194/acp-26-7463-2026, 2026."
        ),
    ),
}


def attribution_for(gcm: str, scenario_group: str) -> Attribution:
    """Return the input attribution for one GCM and scenario group.

    Parameters
    ----------
    gcm : str
        GCM key as the stores spell it, e.g. ``"CESM2-WACCM6"``.
    scenario_group : str
        Scenario group as the stores spell it, e.g. ``"g6_1p5k"``.

    Returns
    -------
    Attribution
        The license and citation for that simulation.

    Raises
    ------
    KeyError
        If the pair is not recorded at all. Publishing data we cannot credit is a bug, so this
        raises rather than falling back to a default.
    """
    try:
        return INPUT_ATTRIBUTION[(gcm, scenario_group)]
    except KeyError:
        raise KeyError(
            f"no license or citation recorded for {gcm!r}/{scenario_group!r}. "
            "Add it to INPUT_ATTRIBUTION and to docs/access-data/licenses.md."
        ) from None


def metadata_attrs(gcm: str, scenario_group: str, *, product: str) -> dict[str, str]:
    """Return the license and attribution attrs for one group.

    Parameters
    ----------
    gcm : str
        GCM key as the stores spell it.
    scenario_group : str
        Scenario group as the stores spell it.
    product : {"output", "input"}
        ``"output"`` is our downscaled product, uniformly :data:`OUTPUT_LICENSE`, and still
        credits the upstream simulation through ``references``. ``"input"`` is the source
        simulation, which carries its own license, or none.

    Returns
    -------
    dict of str to str
        Attribute names and values, using CF and ACDD spellings.

    Notes
    -----
    Two keys are deliberately conditional. ``license`` and ``license_url`` are omitted for an
    input simulation that asserts no license, so that no caller mistakes silence for a grant.
    ``institution`` is set only on output, because ACDD defines it as the producer of the data:
    on an input group that is the modeling center, not us, and several input groups already
    record their own.
    """
    upstream = attribution_for(gcm, scenario_group)
    attrs = {"references": upstream.references}
    if product == "output":
        attrs["license"] = OUTPUT_LICENSE
        attrs["license_url"] = LICENSE_URLS[OUTPUT_LICENSE]
        attrs["institution"] = INSTITUTION
    elif upstream.license is not None:
        attrs["license"] = upstream.license
        attrs["license_url"] = upstream.license_url
    return attrs
