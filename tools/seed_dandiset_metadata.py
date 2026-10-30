"""One-off: build dandi_scraper/dandiset_metadata.csv from the brain regions
already in all_new.csv. Legacy values are mapped through BRAIN_REGION_SYNONYMS;
legacy 'unknown' / missing ones are re-asked with the constrained Cohere prompt.
Every row starts with reviewed=False.

Usage: python tools/seed_dandiset_metadata.py [path/to/all_new.csv]
"""
import os
import sys

import pandas as pd
from dandi.dandiapi import DandiAPIClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dandi_scraper import metadata_table  # noqa: E402
from dandi_scraper._metadata_parser import suggest_brain_region  # noqa: E402
from dandi_scraper.logging_utils import configure_logging  # noqa: E402


def main(merged_csv="all_new.csv"):
    configure_logging()
    if os.path.exists(metadata_table.TABLE_PATH):
        sys.exit(f"{metadata_table.TABLE_PATH} already exists; delete it to reseed")
    df = pd.read_csv(merged_csv, usecols=["dandiset label", "brain_region"], low_memory=False)
    legacy = df.groupby("dandiset label")["brain_region"].first()

    client = DandiAPIClient()
    rows = []
    for did, raw in legacy.items():
        meta = client.get_dandiset(str(int(did)).zfill(6)).get_raw_metadata()
        region, subregion = metadata_table.normalize_region(raw)
        row = dict(dandiset_id=int(did), title=meta["name"], brain_region=region,
                   brain_subregion=subregion, source="llm", confidence="",
                   reviewed=False, notes=f"legacy value: {raw}")
        if region == "Unknown":
            species = (meta["assetsSummary"].get("species") or [{}])[0]
            suggestion = suggest_brain_region(
                did, meta["name"], meta.get("description"), meta.get("keywords"),
                metadata_table.normalize_species(species.get("name"), species.get("identifier")),
            )
            if suggestion is None:
                row.update(source="default", notes="no LLM suggestion available")
            else:
                row.update(suggestion)
        rows.append(row)
        print(f"{did}: {row['brain_region']} / {row['brain_subregion']}  ({row['source']})")

    metadata_table.save_table(pd.DataFrame(rows).set_index("dandiset_id"))
    print(f"wrote {len(rows)} rows to {metadata_table.TABLE_PATH}")


if __name__ == "__main__":
    main(*sys.argv[1:])
