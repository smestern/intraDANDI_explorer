
#for deployment we have to force install some packages here
import sys
import subprocess
#ipfx release fails to install if numpy is not installed first, we have to force install it here
packages = ['ipfx', ]
# implement pip as a subprocess:
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--no-deps'])

#test imports
    import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
from dandi.download import download as dandi_download
from collections import defaultdict
#dandi functions
# os sys imports
from pyAPisolation.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from pyAPisolation.patch_ml import *
import os
import shutil

#import fsspec
#from fsspec.implementations.cached import CachingFileSystem
import glob
import scipy.stats
import joblib
# dash / plotly imports

import plotly.graph_objs as go
from plotly.subplots import make_subplots
# data science imports
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

#local imports
from pyAPisolation.database.build_database import run_analysis, parse_long_pulse_from_dataset, build_dataset_traces

import pyAPisolation.webViz.run_web_viz as wbz
import pyAPisolation.webViz.webVizConfig as wvc
import pyAPisolation.webViz.ephysDatabaseViewer as edb
from ._metadata_parser import dandi_meta_parser

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import umap
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

#global the cache
# FS = CachingFileSystem(
#         fs=fsspec.filesystem("http"),
#         cache_storage="nwb-cache",  # Local folder for the cache
#     )


cols_to_keep = ['input_resistance', 'tau', 'v_baseline', 'sag_nearest_minus_100', 
       'ap_1_threshold_v_0_long_square', 'ap_1_peak_v_0_long_square',
       'ap_1_upstroke_0_long_square',# 'ap_1_downstroke_0_long_square',
       #'ap_1_upstroke_downstroke_ratio_0_long_square',
       'ap_1_width_0_long_square', 'ap_1_fast_trough_v_0_long_square',
       'ap_mean_threshold_v_0_long_square', 'ap_mean_peak_v_0_long_square',
       'ap_mean_upstroke_0_long_square', #'ap_mean_downstroke_0_long_square',
       #'ap_mean_upstroke_downstroke_ratio_0_long_square',
       'ap_mean_width_0_long_square', 'ap_mean_fast_trough_v_0_long_square',
       'avg_rate_0_long_square', 'latency_0_long_square',]




def build_dandiset_df():
    client = DandiAPIClient()

    dandisets = list(client.get_dandisets())

    species_replacement = {
        "Mus musculus - House mouse": "House mouse",
        "Rattus norvegicus - Norway rat": "Rat",
        "Brown rat": "Rat",
        "Rat; norway rat; rats; brown rat": "Rat",
        "Homo sapiens - Human": "Human",
        "Drosophila melanogaster - Fruit fly": "Fruit fly",
    }

    neurodata_type_map = dict(
        ecephys=["LFP", "Units", "ElectricalSeries"],
        ophys=["PlaneSegmentation", "TwoPhotonSeries", "ImageSegmentation"],
        icephys=["PatchClampSeries", "VoltageClampSeries", "CurrentClampSeries"],
    )

    def is_nwb(metadata):
        return any(
            x['identifier'] == 'RRID:SCR_015242'
            for x in metadata['assetsSummary'].get('dataStandard', {})
        )

    data = defaultdict(list)
    for dandiset in dandisets:
        identifier = dandiset.identifier
        metadata = dandiset.get_raw_metadata()
        if not is_nwb(metadata) or not dandiset.draft_version.size:
            continue
        data["identifier"].append(identifier)
        data["created"].append(dandiset.created)
        data["size"].append(dandiset.draft_version.size)
        if "species" in metadata["assetsSummary"] and len(metadata["assetsSummary"]["species"]):
            data["species"].append(metadata["assetsSummary"]["species"][0]["name"])
        else:
            data["species"].append(np.nan)
        
        
        for modality, ndtypes in neurodata_type_map.items():
            data[modality].append(
                any(x in ndtypes for x in metadata["assetsSummary"]["variableMeasured"])
            )
        
        if "numberOfSubjects" in metadata["assetsSummary"]:
            data["numberOfSubjects"].append(metadata["assetsSummary"]["numberOfSubjects"])
        else:
            data["numberOfSubjects"].append(np.nan)

        data['keywords'].append([x.lower() for x in metadata.get("keywords", [])])
    df = pd.DataFrame.from_dict(data)

    for key, val in species_replacement.items():
        df["species"] = df["species"].replace(key, val)
    return df


def get_dandi_metadata(code):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(code)
    metadata_parser = dandi_meta_parser(code)
    metadata = metadata_parser.asset_data

    return metadata

def analyze_dandiset(code, cache_dir=None):
    df_dandiset = run_analysis(cache_dir+'/'+code)
    return df_dandiset
    
def filter_dandiset_df(row, species=None, modality=None, keywords=[], method='or'):
    flags = []
    if species is not None:
        flags.append(row["species"] == species)
    if modality is not None:
        flags.append(row[modality] == True)
    if len(keywords) > 0:
        flags.append(np.any(np.ravel([[x in j for j in row["keywords"]] for x in keywords])))
    if method == 'or':
        return any(flags)
    elif method == 'and':
        return all(flags)
    else:
        raise ValueError("method must be 'or' or 'and'")


def download_dandiset(code=None, save_dir=None, overwrite=False):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(code)
    if save_dir is None:
        save_dir = os.getcwd()
    if os.path.exists(save_dir+'/'+code) and overwrite==False:
        return
    dandi_download(dandiset.api_url, save_dir)
    

def quick_qc(df, qc_features={'input_resistance':[0, 1e9],'sag_nearest_minus_100':[-1, 1],
                             'ap_1_threshold_v_0_long_square':[-100, 100],
                             'tau':[(0.01/1000), (0.4)],
                             'ap_1_width_0_long_square':[(0.01/1000), (10/1000)],
                             'ap_1_fast_trough_v_0_long_square':[-100, 0],
                             'avg_rate_0_long_square':[0, 200]
                             }):
    #this is a quick qc function to check if the data is good, specifically for the output of the analyze_dandiset function
    for feature, (min_val, max_val) in qc_features.items():
        if feature in df.columns:
            #get number of failing values for logging
            _failing = df[(df[feature] < min_val) | (df[feature] > max_val)]
            num_failing = len(_failing)
            print(f"QC: {num_failing} cells failed {feature} check ({min_val} to {max_val}), examples:\n {_failing[feature].head()}")
            df = df[(df[feature] >= min_val) & (df[feature] <= max_val)]

   
    return df

def scale_features(df, features={'input_resistance': 'log', 'tau': 'log-1000', 'ap_1_width_0_long_square': 'log-1000'}):
    for feature, method in features.items():
        if feature in df.columns:
            if method == 'log':
                df[feature] = np.log10(df[feature])
            elif method == 'log-1000':
                df[feature] = np.log10(df[feature]*1000)
            elif method == 'zscore':
                df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
            else:
                raise ValueError("method must be 'log' or 'zscore'")
    return df

dandisets_to_skip = ['000012', '000013', 
'000008', 
'000020',                    
'000005', #mostly in vivo continous data
'000117', '000168', 
'000362', #appears to be some lfp or something
'000717', #test dandiset
 '000293', #Superseeded by 000297
  '000292', #Superseeded by 000297
  '000341' ] #Superseeded by 000297
dandisets_to_include = ['000008', '000035'] #these are iCEphys datasets that are not labeled as such


def run_analyze_dandiset():
    """ 
    Analyze the dandiset and save the results to a csv file, this function will download the dandiset if it is not already downloaded
    then it will analyze the dandiset and save the results to a csv file.
    """
    dandi_df = build_dandiset_df() #pull all the dandisets
    filtered_df = dandi_df[dandi_df.apply(lambda x: filter_dandiset_df(x, modality='icephys', keywords=['intracellular', 'patch'], method='or'), axis=1)] #filter the dandisets we only want the icephys ones
    
    print(f"found {len(filtered_df)} dandisets to analyze")
    #glob the csv files
    csv_files = glob.glob('/media/smestern/Expansion/dandi/*.csv')
    csv_files = [x.split('/')[-1].split('.')[0] for x in csv_files]
    filtered_df = filtered_df[~filtered_df["identifier"].isin(csv_files)] #remove any dandisets we have already analyzed
    #add in the ones we specifically want
    for code in dandisets_to_include:
        if code not in filtered_df["identifier"].values:
            filtered_df = pd.concat([filtered_df, dandi_df[dandi_df["identifier"] == code]])


    for row in filtered_df.iterrows():
        print(f"Downloading {row[1]['identifier']}")
        if row[1]["identifier"] in dandisets_to_skip:
            print(f"Skipping {row[1]['identifier']}")
            continue
        download_dandiset(row[1]["identifier"], save_dir='/media/smestern/Expansion/dandi', overwrite=False)
        df_dandiset = analyze_dandiset(row[1]["identifier"],cache_dir='/media/smestern/Expansion/dandi/')
        df_dandiset["dandiset"] = row[1]["identifier"]
        df_dandiset["created"] = row[1]["created"]
        df_dandiset["species"] = row[1]["species"]
        df_dandiset.to_csv('/media/smestern/Expansion/dandi/'+row[1]["identifier"]+'.csv')

def run_merge_dandiset(use_cached_metadata=True):
    """"""
    csv_files = glob.glob('/media/smestern/Expansion/dandi/*.csv')
    csv_files = [x.split('/')[-1].split('.')[0] for x in csv_files]
    dfs = []
    dataset_numeric = []
    for code in csv_files:
        if code == 'all':
            continue
        temp_df = pd.read_csv('/media/smestern/Expansion/dandi/'+code+'.csv', index_col=0)
        #temp_df = temp_df.dropna(axis=0, how='all', subset=temp_df.columns[:-3])
        temp_df.rename(columns={'dandiset': 'dandiset label', 'species label': 'species'}, inplace=True)
        
        dfs.append(temp_df)
        
    if os.path.exists('./all_new.csv'):
        #load all new for cached metadata stuff
        df_old = pd.read_csv('./all_new.csv', index_col=0)
    else:
        df_old = None
    

    dfs = pd.concat(dfs)
    #remap the dandiset label so that its a string
    dfs['dandiset label'] = dfs['dandiset label'].apply(lambda x: '000000'[:6-len(str(x))]+str(x))
   
    
    #remap the indexes, this is a nightmare due to custom pathing on my local machine
    dfs.index = dfs.index.map(lambda x: ''.join(x.split("dandi//")[1]))
    print(dfs.index[:5])
    dfs['specimen_id'] = dfs.index
    dfs['id_full'] = dfs['dandiset label'] + '/' + dfs['specimen_id']
    dfs = quick_qc(dfs)

    #also drop columns where over 50% of the data is missing
    dfs = dfs.dropna(axis=1, thresh=int(len(dfs)*0.9))

    

    idxs = []
    columns = []
    meta_data = []
    for code in dfs['dandiset label'].unique():
        temp_df = dfs.loc[dfs['dandiset label'] == code]
        print(f"Processing {code}")
        #observe the meta data
        if use_cached_metadata and df_old is not None: #If we have old metadata use it
            meta_ = df_old.loc[df_old['dandiset_id'] == int(code), ['dandiset_id', 'age', 'subject_id', 'cell_id', 'brain_region', 'species', 'filepath', 'contributor']]
        else:
            meta_ = get_dandi_metadata(code) #get the meta data from dandi, uses an LLM to parse
        meta_data.append(meta_)
  

        data_num = temp_df.select_dtypes(include=np.number).dropna(axis=1, how='all')
        #data_num should have the same number of rows
        assert len(data_num) == len(temp_df)
        temp_data_num = data_num.copy()
        if data_num.empty or len(data_num.columns) < 3:
            continue #skip empty datasets or ones with too few numeric columns
        
        idxs.append(data_num.index.values)
        print(f"Processing {code} with {len(data_num)} cells")
        #turn nans and infs into nans
        data_num = np.nan_to_num(data_num, nan=np.nan, posinf=np.nan, neginf=np.nan)
        impute = KNNImputer(keep_empty_features=True)
        data_num = impute.fit_transform(data_num)
        print(f"Imputed {code} with {len(data_num)} cells and {len(data_num[0])} features")
        #clip to the 99.5% percentile
        for i in range(data_num.shape[1]):
            col = data_num[:, i] #get the column
            lower_bound = np.nanpercentile(col, 0.5)
            upper_bound = np.nanpercentile(col, 99.5)
            col = np.clip(col, lower_bound, upper_bound)
            data_num[:, i] = col
        

        data_num = pd.DataFrame(data_num, columns=temp_data_num.columns, index=temp_data_num.index)
        print(f"Processed {code} with {len(data_num)} cells")
        assert len(data_num) == len(temp_df)
        dataset_numeric.append(data_num)
    meta_data = pd.concat(meta_data, axis=0)
    print(f"Meta data shape: {meta_data.shape}")
    print(meta_data.head())

    # Merge the meta data
    dfs = dfs.join(meta_data, how='left', rsuffix='_meta')
    print(f"dfs shape after joining meta data: {dfs.shape}")

    # Filter dfs to only include rows where the data is present
    dfs = dfs.loc[np.hstack(idxs)]
    print(f"dfs shape after filtering: {dfs.shape}")

    dataset_numeric = pd.concat(dataset_numeric, axis=0)[cols_to_keep]
    print(f"dataset_numeric shape after concatenation: {dataset_numeric.shape}")
    #log scale some features
    dataset_numeric = scale_features(dataset_numeric, features={'input_resistance': 'log', 'tau': 'log-1000', 'ap_1_width_0_long_square': 'log-1000', 'ap_mean_width_0_long_square': 'log-1000'}) #log scale some features

    #KNN impute again to be on final dataset
    impute = KNNImputer(keep_empty_features=True)
    dataset_numeric.loc[:, :] = impute.fit_transform(dataset_numeric.values) #shoudl work unless we lose a column or row

    # Drop columns where over 50% of the data is missing
    dataset_numeric = dataset_numeric.dropna(axis=1, how='any')
    print(f"dataset_numeric shape after dropping columns: {dataset_numeric.shape}")
    # Ensure dfs and dataset_numeric have the same index
    dfs = dfs.loc[dataset_numeric.index]
    
    #dump the data
    joblib.dump(dataset_numeric, './dataset_numeric.pkl')
    #robust scaler
    scaler = RobustScaler()
    dataset_numeric.loc[:, :] = scaler.fit_transform(dataset_numeric.values)
    
    reducer = umap.UMAP(densmap=False, min_dist=0.3, spread=10, metric='cosine',n_neighbors=500,verbose=True,)

    embedding = reducer.fit_transform(dataset_numeric)
    #also make a n=25 umap
    reducer2 = umap.UMAP(n_neighbors=150, min_dist=0.1, spread=2, repulsion_strength=5, metric='cosine')
    reducer3 = reducer2.fit(dataset_numeric) #+ reducer

    dfs['umap X'] = reducer2.embedding_[:,0]
    dfs['umap Y'] = reducer2.embedding_[:,1]
    plt.scatter(dfs['umap X'], dfs['umap Y'] , s=0.1)

    
    #also run PCA for fun
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(dataset_numeric.fillna(0))
    dfs['pca X'] = embedding[:,0]
    dfs['pca Y'] = embedding[:,1]


    #generate some labels for fun
    gmm = GaussianMixture(n_components=20)
    gmm.fit(dataset_numeric)
    dfs['GMM cluster label'] = gmm.predict(dataset_numeric)

    #add in a supervised umap
    reducer3 = umap.UMAP(densmap=False, target_weight=0.1, n_neighbors=50, verbose=True,)
    embedding = reducer3.fit(dataset_numeric, y=dfs['GMM cluster label']) + reducer2
    dfs['supervised umap X'] = reducer3.embedding_[:,0]
    dfs['supervised umap Y'] = reducer3.embedding_[:,1]
    #plot it


    
    plt.scatter(dfs['umap X'], dfs['umap Y'] , s=0.1
                
                )
    
    #plt.show()
    dfs["dandiset_link"] = dfs["dandiset label"].apply(lambda x: f"https://dandiarchive.org/dandiset/{str(int(x)).zfill(6)}")
    file_link = []
    meta_data_link = []
    with DandiAPIClient() as client:
        for dandiset_id, specimen_id in zip(dfs['dandiset label'], dfs['specimen_id']):
            if use_cached_metadata and df_old is not None:
                
                asset = df_old.loc[df_old['id_full'] == (dandiset_id + '/' + specimen_id)]
                if asset.empty: #not found in old df
                    asset = client.get_dandiset(str(int( dandiset_id)).zfill(6), 'draft').get_asset_by_path('/'.join(specimen_id.split('/')[1:]))
                    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                    meta_data_link.append(asset.api_url)
                    file_link.append(s3_url)
                    continue
                s3_url = asset['file_link'].values[0]
                meta_data_link.append(asset['meta_data_link'].values[0])
            else:
                asset = client.get_dandiset(str(int( dandiset_id)).zfill(6), 'draft').get_asset_by_path('/'.join(specimen_id.split('/')[1:]))
                s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                meta_data_link.append(asset.api_url)
            file_link.append(s3_url)
    dfs['file_link'] = file_link
    dfs['meta_data_link'] = meta_data_link
    
    
    dfs.to_csv('./all_new.csv')


def run_plot_dandiset():
    csv_files = glob.glob('/media/smestern/Expansion/dandi/*.csv')
    csv_files = [x.split('/')[-1].split('.')[0] for x in csv_files]
    for code in csv_files:
        #find the folder
        #load the csv so we can filter ids
        df = pd.read_csv('/media/smestern/Expansion/dandi/'+code+'.csv', index_col=0)
        ids = [x.split("/dandi/")[-1] for x in df.index.values]
        print(f"Processing {code}")
        #if code != "000142":
        #continue
        if code in dandisets_to_skip:
            print(f"Skipping {code}")
            continue
        if code == 'all':
            print(f"Skipping {code}")
            continue
        folder = f"/media/smestern/Expansion/dandi/{code}"
        
        build_dataset_traces(folder, ids, parallel=True)
    
def sort_plot_dandiset():
    svg_files = glob.glob('/media/smestern/Expansion/dandi/**/*.svg', recursive=True)
    for svg_file in svg_files:
        print(f"Processing {svg_file}")
        #split and save into data/traces
        svg_file_no_dandi = ''.join(svg_file.split('/dandi/')[-1])
        #move into the local folder
        local_path = "./data/traces/"+svg_file_no_dandi
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.move(svg_file, local_path)
    return



def build_server():
    GLOBAL_STIM_NAMES.stim_inc =['']
    GLOBAL_VARS = wvc.webVizConfig()
    GLOBAL_VARS.file_index = 'specimen_id'
    GLOBAL_VARS.file_path = 'specimen_id'
    GLOBAL_VARS.table_vars_rq = ['specimen_id', 'ap_1_width_0_long_square', 'input_resistance','tau','v_baseline',
                                 'sag_nearest_minus_100', 'ap_1_threshold_v_0_long_square', 'ap_1_peak_v_0_long_square', 'file_link', 'dandiset_link', "meta_data_link"]
    GLOBAL_VARS.table_vars = [ 'input_resistance','tau','v_baseline','sag_nearest_minus_100', 'species', 'brain_region',]#
    GLOBAL_VARS.para_vars = ['ap_1_width_0_long_square', 'input_resistance','tau','v_baseline','sag_nearest_minus_100', 'species', 'brain_region']
    GLOBAL_VARS.para_var_colors = 'ap_1_width_0_long_square'
    GLOBAL_VARS.umap_labels = ['dandiset label', 'species', 'brain_region', 'contributor',
                                {'Ephys Feat:': 
cols_to_keep }]#['input_resistance','tau','v_baseline','sag_nearest_minus_100', 
                                                 #'ap_1_width_0_long_square']}]
    GLOBAL_VARS.plots_path = '.'
    #GLOBAL_VARS.primary_label = 'dandiset label'
    #GLOBAL_VARS.primary_label = 'brain_region'
    GLOBAL_VARS.umap_cols = ['umap X', 'umap Y']
    GLOBAL_VARS.hidden_table = True
    GLOBAL_VARS.hidden_table_vars = ['dandiset label', 'species']
    #Add a title to the webviz
    GLOBAL_VARS.db_title = "Icephys Dandiset Visualization"
    GLOBAL_VARS.db_description = """ This is a visualization of some of the intracellular electrophysiology (icephys) data found across the open \n
    neuroscience initiative DANDI. The data is visualized using a UMAP and a parallel coordinates plot. The data is also visualized in a table format. \n
    This is currently a work in progress and is not yet complete. Please cite the original authors of the data when using this data. """
    GLOBAL_VARS.db_subtitle = ""
    GLOBAL_VARS.db_links = {'Dandi': 'https://dandiarchive.org/',  "smestern on X": "https://twitter.com/smestern"}
    GLOBAL_VARS.db_para_title = "Paracoords"
    GLOBAL_VARS.db_embed_title = "UMAP"

    GLOBAL_VARS.col_rename = {
    "ap_1_width_0_long_square": "Rheo-AP width Log[(ms)]",
    "sag_nearest_minus_100": "Sag",
    "input_resistance": "Input resistance Log[(MOhm)]",
    "tau": "Tau Log[(ms)]",
    "ap_1_threshold_v_0_long_square": "Rheo-AP Threshold (mV)",
    "ap_1_peak_v_0_long_square": "Rheo-AP Peak (mV)",
    "ap_1_upstroke_0_long_square": "Rheo-AP Upstroke (mV/ms)",
    "ap_1_fast_trough_v_0_long_square": "Rheo-AP Fast Trough (mV)",
    "ap_mean_threshold_v_0_long_square": "Mean AP Threshold (mV)",
    "ap_mean_peak_v_0_long_square": "Mean AP Peak (mV)",
    "ap_mean_upstroke_0_long_square": "Mean AP Upstroke (mV/ms)",
    "ap_mean_width_0_long_square": "Mean AP Width Log[(ms)]",
    "ap_mean_fast_trough_v_0_long_square": "Mean AP Fast Trough (mV)",
    "avg_rate_0_long_square": "Avg Firing Rate (Hz)",
    "latency_0_long_square": "Latency (s)",
    "v_baseline": "Baseline voltage (mV)",
    "dandiset_link": "View Dandiset",
    "meta_data_link": "View File Metadata",
    "file_link": "File Download",
    }


    GLOBAL_VARS.table_spec = {"View Dandiset": "links", "View File Metadata": "links", "File Download": "links"}

    # GLOBAL_VARS.table_split = 'species'
    # GLOBAL_VARS.split_default = "Human"
    filepath = os.path.dirname(os.path.abspath(__file__))

    #load the data
    file = pd.read_csv(filepath+'/../all_new.csv',)

    file["ap_1_width_0_long_square"] = np.log10(file["ap_1_width_0_long_square"]*1000)
    #do the same for the tau
    file["tau"] = np.log10(file["tau"]*1000)
    file['input_resistance'] = np.log10(file['input_resistance'])

    #shuffle the data for fun
    file = file.sample(frac=1)

    file.to_csv(filepath+'/../all2.csv')

    wbz.run_web_viz(database_file=filepath+'/../all2.csv', config=GLOBAL_VARS, backend='static')
    return
    
if __name__ == "__main__":
    build_server()