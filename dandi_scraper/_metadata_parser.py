
#try:
#load cohere ai for parsing descriptions
import cohere
from . import config
co = cohere.Client(config.COHERE_KEY)
#except:
#co = None
print("cohere.ai not installed; skipping description parsing")

import time
from dandi.dandiapi import DandiAPIClient
from dandi.download import download as dandi_download
import pandas as pd


EXAMPLE_PROMPT = """ Title: Phenotypic variation within and across transcriptomic cell types in mouse motor cortex
                    Description: We used Patch-seq to combine patch-clamp recording, biocytin staining, and single-cell RNA sequencing of over 1300 neurons in adult mouse motor cortex, providing a comprehensive morpho-electric annotation of almost all transcriptomically defined neural cell types. Contained in this dandiset are the intracellular electrophysiological recordings. See Dandiset #35 for an additional dataset, recorded under the physiological temperature.
                    Keywords: Patch-seq, cortex, motor cortex, mouse
                    Species: Mus musculus
                    In one to two words, what brain region(s) were studied?: Motor cortex
                    
                    /n
                    
                    Title: IVSCC stimulus sets
                    Description: Allen Institute for Brain Science IVSCC (In-vitro Single Cell Characterization) project stimulus sets stored in NWB format
                    Keywords: 
                    Species:
                    In one to two words, what brain region(s) were studied?: Unknown"""




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

        self.dandiset_species = self.metadata['assetsSummary']['species'] if 'species' in self.metadata['assetsSummary'] else None
        self.dandiset_brain_region = self.determine_brain_region()
        self.asset_data = self.build_asset_data()
        print(f"Metadata parsed for dandiset {self.dandiset_id}, found {len(self.asset_data)} assets")
        print(f"Brain region: {self.dandiset_brain_region}")

        
    def build_asset_data(self):
        asset_data = []
        for asset in self.dandiset.get_assets():
            meta_asst = asset.get_metadata()
            try:
                #figure out the age
                age = pd.Timedelta(meta_asst.wasAttributedTo[0].age.value).days
                subject_id = meta_asst.wasAttributedTo[0].id if meta_asst.wasAttributedTo[0].id != None else meta_asst.wasAttributedTo[0].identifier
                cell_id = meta_asst.wasGeneratedBy[0].id if meta_asst.wasGeneratedBy[0].id != None else meta_asst.wasGeneratedBy[0].identifier
                species = meta_asst.wasAttributedTo[0].species.name if meta_asst.wasAttributedTo[0].species != None else self.dandiset_species
                filepath = meta_asst.path

                #make the index a concat of the dandiset_id and the filepath
                idxs = self.dandiset_id + '/' + filepath
                asset_data.append({'idxs':idxs, 'dandiset_id': self.dandiset_id, 'age': age, 'subject_id': subject_id, 
                'cell_id': cell_id, 'brain_region': self.dandiset_brain_region, 'species': species, 'filepath': filepath, 'contributor': self.first_contributor})
            except:
                asset_data.append({'idxs': self.dandiset_id + '/' + meta_asst.path, 'dandiset_id': self.dandiset_id, 'age': None, 'subject_id': None,
                'cell_id': None, 'brain_region': self.dandiset_brain_region, 'species': self.dandiset_species, 'filepath': meta_asst.path, 'contributor': self.first_contributor})
        return pd.DataFrame.from_dict(asset_data).set_index('idxs')

    def determine_brain_region(self):
        PROMPT = """
                    Title: {}
                    Description: {}
                    Keywords: {}
                    Species: {}
                    What brain region(s) were studied (in one to two words)?:""".format(self.dandiset_name, self.dandiset_description, self.dandiset_keywords, self.dandiset_species)
        extraction = co.chat(
           message=EXAMPLE_PROMPT + '\n' + PROMPT,)
        #time.sleep(60)
        text = extraction.text
        #sometimes the model returns the prompt included so we want to split at the colon and take the second part
        if ':' in text:
            text = ''.join(text.split(':')[-1])
        return text.strip()
