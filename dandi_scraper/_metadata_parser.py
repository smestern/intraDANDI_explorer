"""Per-asset metadata for a dandiset. The brain region comes from
`dandiset_metadata.csv`; Cohere only suggests rows for dandisets missing from it.
"""
import json
import logging

import pandas as pd
from dandi.dandiapi import DandiAPIClient

from . import config
from .metadata_table import lookup_or_add, normalize_species

logger = logging.getLogger("dandi_scraper.metadata")


def _make_cohere_client():
    try:
        import cohere
    except ImportError:
        logger.warning("cohere not installed; new dandisets cannot get an LLM brain-region suggestion")
        return None
    try:
        return cohere.ClientV2(api_key=config.get_cohere_key())
    except RuntimeError as exc:
        logger.warning("%s New dandisets cannot get an LLM brain-region suggestion.", exc)
        return None


co = _make_cohere_client()


SYSTEM_PROMPT = f"""You are a neuroscience metadata extraction expert. Your task is to analyze scientific dataset metadata and identify the brain region studied.

Based on the provided metadata (title, description, keywords, and species), return JSON with:
- "brain_region": exactly one of {config.BRAIN_REGIONS}
- "brain_subregion": a short, lowercase, standard anatomical name for the specific area (e.g. "motor cortex", "dentate gyrus", "L1"), or "" if not stated
- "confidence": "high", "medium" or "low"
- "reasoning": brief explanation

Guidelines:
- All cortical areas (motor, visual, temporal, insular, ...) are "Neocortex"; put the area in brain_subregion
- If multiple regions, use the primary one
- If the region is not stated in the metadata, use "Unknown" with "low" confidence. Do not guess from the lab or species.
"""

EXAMPLE_PROMPT = """Example 1:
Title: Phenotypic variation within and across transcriptomic cell types in mouse motor cortex
Description: We used Patch-seq to combine patch-clamp recording, biocytin staining, and single-cell RNA sequencing of over 1300 neurons in adult mouse motor cortex, providing a comprehensive morpho-electric annotation of almost all transcriptomically defined neural cell types. Contained in this dandiset are the intracellular electrophysiological recordings.
Keywords: Patch-seq, cortex, motor cortex, mouse
Species: Mus musculus

Response:
{"brain_region": "Neocortex", "brain_subregion": "motor cortex", "confidence": "high", "reasoning": "Title and description explicitly mention motor cortex"}

Example 2:
Title: IVSCC stimulus sets
Description: Allen Institute for Brain Science IVSCC (In-vitro Single Cell Characterization) project stimulus sets stored in NWB format
Keywords:
Species:

Response:
{"brain_region": "Unknown", "brain_subregion": "", "confidence": "low", "reasoning": "No brain region mentioned in available metadata"}

Example 3:
Title: Excitability of dentate granule cells after seizures
Description: Whole-cell recordings from dentate gyrus granule cells in acute hippocampal slices from adult mice
Keywords: hippocampus, dentate gyrus, patch clamp
Species: Mus musculus

Response:
{"brain_region": "Hippocampus", "brain_subregion": "dentate gyrus", "confidence": "high", "reasoning": "Description states dentate gyrus granule cells in hippocampal slices"}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "brain_region": {"type": "string", "enum": config.BRAIN_REGIONS},
        "brain_subregion": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
    },
    "required": ["brain_region", "brain_subregion", "confidence", "reasoning"],
}


def suggest_brain_region(dandiset_id, title, description, keywords, species):
    """Ask Cohere for a brain region constrained to `config.BRAIN_REGIONS`.

    Returns a row dict for `metadata_table`, or None if the LLM is
    unavailable or the call fails.
    """
    if co is None:
        logger.error(
            "dandiset %s is not in dandiset_metadata.csv and no Cohere client is "
            "available; brain_region will be 'Unknown'", dandiset_id,
        )
        return None
    query = f"""Now analyze this dataset:
Title: {title}
Description: {description}
Keywords: {keywords}
Species: {species}

Response:"""
    try:
        response = co.chat(
            model=config.COHERE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{EXAMPLE_PROMPT}\n\n{query}"},
            ],
            response_format={"type": "json_object", "json_schema": RESPONSE_SCHEMA},
            temperature=0,
        )
        result = json.loads(response.message.content[0].text)
    except Exception as exc:
        logger.error("Cohere brain-region suggestion failed for %s: %s", dandiset_id, exc)
        return None
    logger.debug("Cohere response for %s: %s", dandiset_id, result)
    return {
        "brain_region": result["brain_region"],
        "brain_subregion": result["brain_subregion"].strip().lower(),
        "confidence": result["confidence"],
        "notes": result["reasoning"],
    }


class dandi_meta_parser():
    def __init__(self, dandiset_id):
        self.dandiset_id = dandiset_id
        self.client = DandiAPIClient("https://api.dandiarchive.org/api/")
        self.dandiset = self.client.get_dandiset(dandiset_id)
        self.metadata = self.dandiset.get_raw_metadata()
        self.dandiset_name = self.metadata['name']
        self.dandiset_description = self.metadata['description'] if 'description' in self.metadata else None
        self.dandiset_keywords = self.metadata['keywords'] if 'keywords' in self.metadata else None
        self.first_contributor = self.metadata['contributor'][0]['name']

        species_list = self.metadata['assetsSummary'].get('species') or []
        self.dandiset_species = (
            normalize_species(species_list[0].get('name'), species_list[0].get('identifier'))
            if species_list else None
        )
        self.dandiset_brain_region, self.dandiset_brain_subregion = lookup_or_add(
            self.dandiset_id, self.dandiset_name,
            lambda: suggest_brain_region(
                self.dandiset_id, self.dandiset_name, self.dandiset_description,
                self.dandiset_keywords, self.dandiset_species,
            ),
        )
        self.asset_data = self.build_asset_data()
        logger.info(
            "metadata parsed for dandiset %s: %d assets, brain region %s / %s",
            self.dandiset_id, len(self.asset_data),
            self.dandiset_brain_region, self.dandiset_brain_subregion,
        )

    def build_asset_data(self):
        asset_data = []
        for asset in self.dandiset.get_assets():
            meta_asst = asset.get_metadata()
            row = {'idxs': self.dandiset_id + '/' + meta_asst.path, 'dandiset_id': self.dandiset_id,
                   'age': None, 'subject_id': None, 'cell_id': None,
                   'brain_region': self.dandiset_brain_region,
                   'brain_subregion': self.dandiset_brain_subregion,
                   'species': self.dandiset_species, 'filepath': meta_asst.path,
                   'contributor': self.first_contributor}
            try:
                subject = meta_asst.wasAttributedTo[0]
                if subject.species is not None:
                    row['species'] = normalize_species(subject.species.name, subject.species.identifier)
                row['subject_id'] = subject.id if subject.id is not None else subject.identifier
                cell_id = meta_asst.wasGeneratedBy[0].id if meta_asst.wasGeneratedBy[
                    0].id != None else meta_asst.wasGeneratedBy[0].identifier
                row['cell_id'] = cell_id
                row['age'] = pd.Timedelta(subject.age.value).days
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                logger.debug("dandiset %s asset %s: incomplete metadata (%s)", self.dandiset_id, meta_asst.path, exc)
            asset_data.append(row)
        return pd.DataFrame.from_dict(asset_data).set_index('idxs')
