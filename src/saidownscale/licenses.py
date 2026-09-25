"""Canonical license and attribution for the SAI Downscaling inputs and output.

This module is the single source of truth. The store attrs written by
:mod:`saidownscale.apply_store_metadata` and the table in ``docs/access-data/licenses.md`` derive
from it, and ``tests/test_terms_of_data_access.py`` fails if the 2 disagree. Edit here first,
then re-sync the docs table. The ``LICENSE.txt`` files published beside the data on Source
Cooperative are not generated from this module yet.

Licenses are named exactly as ``docs/access-data/licenses.md`` names them, so the published data
and the published table never disagree, and each one pairs with a ``license_url`` that resolves it.
A license of ``None`` means we assert no license for that simulation, which is different from not
knowing: the attribution is still recorded, and nothing downstream may invent one in its place.

``None`` is published as an empty string rather than as a missing attribute, matching the blank cell
in the docs table. An absent attribute reads as an oversight, while a blank one shows the question
was asked and answered, and neither can be mistaken for a grant.
"""

from __future__ import annotations

from dataclasses import dataclass

CC_BY_4_0 = "CC BY 4.0"
OGL_V3 = "OGLv3"

#: Every license we name, mapped to the text that governs it. The keys double as the set of
#: spellings the docs table may use, so a stale spelling there fails rather than passing quietly.
LICENSE_URLS: dict[str, str] = {
    CC_BY_4_0: "https://creativecommons.org/licenses/by/4.0/",
    OGL_V3: "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
}

#: Our downscaled output carries one license, whatever it was derived from. The upstream
#: simulation still has to be credited, which is what the citation below is for.
OUTPUT_LICENSE = CC_BY_4_0

INSTITUTION = "CarbonPlan"

#: Where to write about the data. Set on our own output only, alongside ``institution``.
CONTACT = "hello@carbonplan.org"

#: The Zenodo record for this work, published on our output only. Stored as a resolvable URL, so
#: it matches ``license_url`` and ``terms_of_data_access`` beside it and a reader can follow it
#: without knowing to add a prefix. Input groups do not get it: they carry their own upstream
#: identifiers inside ``attribution``, and ours would credit the wrong producer.
DOI = "https://doi.org/10.5281/zenodo.22932138"

#: Where we publish the Terms of Data Access. Set on both products, because the terms govern the
#: input simulations we redistribute as much as the output we derive from them. Read the Docs serves
#: this project multi-version, so the ``/en/latest/`` segment is part of the URL rather than
#: optional: without it the page 404s.
TERMS_OF_DATA_ACCESS = (
    "https://sai-downscaling.readthedocs.io/en/latest/access-data/terms-of-data-access.html"
)


@dataclass(frozen=True)
class Attribution:
    """The license and citation for one input simulation.

    Parameters
    ----------
    license : str or None
        License name as the docs table spells it, e.g. ``"CC BY 4.0"``. ``None`` where we assert
        no license.
    references : str
        Citation text, published as ``attribution`` on an input group. Our own output cites its
        ``doi`` instead, whose record credits every input.
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
            "https://doi.org/10.22033/ESGF/CMIP6.10071 Note: CC BY-SA 4.0 International "
            "License is named in the netCDF file; we understand this licensing is superseded "
            "by CC BY 4.0, based on the CMIP6 Terms of Use."
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
            "and G6sulfur: Changes in impacts and uncertainty depending on stratospheric "
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
        license=OGL_V3,
        references=(
            "These simulations were run by Andy Jones, in collaboration with Jim Haywood and "
            "Matthew Henry, and provided by Matthew Henry. The UK Earth System Model is "
            "developed and maintained by the Met Office Hadley Centre."
        ),
    ),
    ("UKESM1-1-LL", "g6_1p5k"): Attribution(
        license=OGL_V3,
        references=(
            "These simulations were run by Andy Jones, in collaboration with Jim Haywood and "
            "Matthew Henry, and provided by Matthew Henry. The UK Earth System Model is "
            "developed and maintained by the Met Office Hadley Centre. See also: Lee, W. R., "
            "Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, "
            "T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M.: G6-1.5K-SAI and "
            "G6sulfur: Changes in impacts and uncertainty depending on stratospheric aerosol "
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
        credits the upstream simulation. ``"input"`` is the source simulation, which carries its
        own license, or none.

    Returns
    -------
    dict of str to str
        Attribute names and values, using CF and ACDD spellings.

    Notes
    -----
    The product decides which keys appear. Input groups publish the upstream citation as
    ``attribution``, the name the licenses table uses. Output groups carry no citation of their own
    inputs, since a scenario leaf is built from several simulations; they point to our ``doi``,
    whose record cites them all. ``institution`` and ``contact`` are set only on output, because
    ACDD defines ``institution`` as the producer of the data, and on an input group that is the
    modeling center, not us. Every group gets a license, a license URL, and the terms, since a
    redistributed simulation needs governing as much as our own product does.
    """
    # Looked up for output too, so a scenario with no recorded provenance is never published.
    upstream = attribution_for(gcm, scenario_group)
    attrs = {"terms_of_data_access": TERMS_OF_DATA_ACCESS}
    if product == "output":
        attrs["license"] = OUTPUT_LICENSE
        attrs["license_url"] = LICENSE_URLS[OUTPUT_LICENSE]
        attrs["institution"] = INSTITUTION
        attrs["contact"] = CONTACT
        # Written only once set. A placeholder in published metadata is worse than no attribute,
        # because a reader cannot tell a stand-in from a real identifier.
        if DOI:
            attrs["doi"] = DOI
    else:
        attrs["attribution"] = upstream.references
        # Blank where we assert no license, never absent. The pair stays whole either way, so a
        # reader never finds a URL resolving a license the group does not name.
        attrs["license"] = upstream.license or ""
        attrs["license_url"] = upstream.license_url or ""
    return attrs
