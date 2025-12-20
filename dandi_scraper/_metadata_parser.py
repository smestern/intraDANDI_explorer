
# try:
# load cohere ai for parsing descriptions
import pandas as pd
from dandi.download import download as dandi_download
from dandi.dandiapi import DandiAPIClient
import time
import json
import cohere
from . import config
co = cohere.Client(config.COHERE_KEY)
# except:
# co = None
# print("cohere.ai not installed; skipping description parsing")


SYSTEM_PROMPT = """You are a neuroscience metadata extraction expert. Your task is to analyze scientific dataset metadata and extract key information about brain regions studied.

Based on the provided metadata (title, description, keywords, and species), identify the brain region(s) studied in the research.

Return ONLY valid JSON in this exact format:
{
  "brain_region": "region name (1-3 words)",
  "confidence": "high/medium/low",
  "reasoning": "brief explanation"
}

Guidelines:
- Use standard anatomical terminology (e.g., "motor cortex", "hippocampus", "visual cortex")
- If multiple regions, list the primary one
- If unclear or not specified, use "unknown" for brain_region with "low" confidence
- Keep brain_region concise (1-3 words maximum)
- Be specific when possible (e.g., "primary visual cortex" rather than just "cortex")
"""

EXAMPLE_PROMPT = """Example 1:
Title: Phenotypic variation within and across transcriptomic cell types in mouse motor cortex
Description: We used Patch-seq to combine patch-clamp recording, biocytin staining, and single-cell RNA sequencing of over 1300 neurons in adult mouse motor cortex, providing a comprehensive morpho-electric annotation of almost all transcriptomically defined neural cell types. Contained in this dandiset are the intracellular electrophysiological recordings.
Keywords: Patch-seq, cortex, motor cortex, mouse
Species: Mus musculus

Response:
{
  "brain_region": "motor cortex",
  "confidence": "high",
  "reasoning": "Title and description explicitly mention motor cortex as the study region"
}

Example 2:
Title: IVSCC stimulus sets
Description: Allen Institute for Brain Science IVSCC (In-vitro Single Cell Characterization) project stimulus sets stored in NWB format
Keywords: 
Species:

Response:
{
  "brain_region": "unknown",
  "confidence": "low",
  "reasoning": "No specific brain region mentioned in available metadata"
}

Example 3:
Title: Visual responses in mouse V1
Description: Single unit recordings from layer 2/3 pyramidal neurons in primary visual cortex of awake mice during visual stimulation
Keywords: visual cortex, V1, electrophysiology
Species: Mus musculus

Response:
{
  "brain_region": "primary visual cortex",
  "confidence": "high",
  "reasoning": "Description explicitly states primary visual cortex (V1) as recording location"
}"""


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

        self.dandiset_species = self.metadata['assetsSummary'][
            'species'] if 'species' in self.metadata['assetsSummary'] else None
        self.dandiset_brain_region = self.determine_brain_region()
        self.asset_data = self.build_asset_data()
        print(
            f"Metadata parsed for dandiset {self.dandiset_id}, found {len(self.asset_data)} assets")
        print(f"Brain region: {self.dandiset_brain_region}")

    def build_asset_data(self):
        asset_data = []
        for asset in self.dandiset.get_assets():
            meta_asst = asset.get_metadata()
            try:
                # figure out the age
                age = pd.Timedelta(meta_asst.wasAttributedTo[0].age.value).days
                subject_id = meta_asst.wasAttributedTo[0].id if meta_asst.wasAttributedTo[
                    0].id != None else meta_asst.wasAttributedTo[0].identifier
                cell_id = meta_asst.wasGeneratedBy[0].id if meta_asst.wasGeneratedBy[
                    0].id != None else meta_asst.wasGeneratedBy[0].identifier
                species = meta_asst.wasAttributedTo[0].species.name if meta_asst.wasAttributedTo[
                    0].species != None else self.dandiset_species
                filepath = meta_asst.path

                # make the index a concat of the dandiset_id and the filepath
                idxs = self.dandiset_id + '/' + filepath
                asset_data.append({'idxs': idxs, 'dandiset_id': self.dandiset_id, 'age': age, 'subject_id': subject_id,
                                   'cell_id': cell_id, 'brain_region': self.dandiset_brain_region, 'species': species, 'filepath': filepath, 'contributor': self.first_contributor})
            except:
                asset_data.append({'idxs': self.dandiset_id + '/' + meta_asst.path, 'dandiset_id': self.dandiset_id, 'age': None, 'subject_id': None,
                                   'cell_id': None, 'brain_region': self.dandiset_brain_region, 'species': self.dandiset_species, 'filepath': meta_asst.path, 'contributor': self.first_contributor})
        return pd.DataFrame.from_dict(asset_data).set_index('idxs')

    def determine_brain_region(self):
        """Extract brain region from metadata using AI with structured JSON output."""
        # Build the query prompt
        query = f"""Now analyze this dataset:
Title: {self.dandiset_name}
Description: {self.dandiset_description}
Keywords: {self.dandiset_keywords}
Species: {self.dandiset_species}

Response:"""
        
        # Combine system prompt, examples, and query
        full_prompt = f"{SYSTEM_PROMPT}\n\n{EXAMPLE_PROMPT}\n\n{query}"
        
        try:
            # Call Cohere API
            extraction = co.chat(
                message=full_prompt,
                temperature=0.3,  # Lower temperature for more consistent output
            )
            
            response_text = extraction.text.strip()
            
            # Try to parse JSON from response
            # Sometimes the model adds text before/after JSON, so find the JSON block
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                brain_region = result.get('brain_region', 'unknown')
                confidence = result.get('confidence', 'unknown')
                reasoning = result.get('reasoning', '')
                
                # Log the structured result
                log_entry = {
                    'dandiset_id': self.dandiset_id,
                    'title': self.dandiset_name,
                    'brain_region': brain_region,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'raw_response': response_text
                }
                
                with open('prompt.txt', 'a') as f:
                    f.write(json.dumps(log_entry, indent=2) + '\n\n')
                
                return brain_region
            else:
                # Fallback: no JSON found, use raw text
                print(f"Warning: Could not parse JSON from response for {self.dandiset_id}")
                with open('prompt.txt', 'a') as f:
                    f.write(f"PARSE ERROR for {self.dandiset_id}:\n{response_text}\n\n")
                return response_text.strip()
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for {self.dandiset_id}: {e}")
            with open('prompt.txt', 'a') as f:
                f.write(f"JSON ERROR for {self.dandiset_id}: {e}\n{response_text}\n\n")
            return "unknown"
        except Exception as e:
            print(f"Error determining brain region for {self.dandiset_id}: {e}")
            with open('prompt.txt', 'a') as f:
                f.write(f"ERROR for {self.dandiset_id}: {e}\n\n")
            return "unknown"
