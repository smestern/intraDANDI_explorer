# %% import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from umap import UMAP

from dandi_scraper import config
from dandi_scraper.analysis import scale_features
from dandi_scraper.database import _drop_duplicate_cells

# %% load the merged database
df = pd.read_csv('all_new.csv', low_memory=False)
df = df.dropna(subset=['umap X']).reset_index(drop=True)
print(df.columns[-20:])
# The features the merge feeds to UMAP / clustering
cols_to_keep = list(config.COLS_TO_KEEP)

# Drop re-released copies of the same recording (same NWB file in several
# dandisets). The merge does this too; this catches CSVs built before it did.
df['dandiset label'] = df['dandiset label'].astype(str).str.zfill(6)
n_before = len(df)
df = _drop_duplicate_cells(df, cols_to_keep, {})
print(f'dropped {n_before - len(df)} duplicate cells, {len(df)} left')


# %% rebuild the matrix the merge feeds to UMAP
def pipeline_features(df, cols):
    """Mirror run_merge_dandiset's preprocessing (dandi_scraper/database.py):
    per-dandiset KNN impute + 0.5-99.5 percentile clip, SCALE_FEATURES log
    transforms, global KNN impute, RobustScaler clipped to +/- ROBUST_CLIP.

    Approximation: the merge imputes each dandiset using all of its numeric
    columns as neighbours; here only ``cols`` are used.
    """
    parts = []
    for _, grp in df[cols].groupby(df['dandiset label']):
        grp = grp.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        vals = KNNImputer(keep_empty_features=True).fit_transform(grp.values)
        lo, hi = np.nanpercentile(vals, [0.5, 99.5], axis=0)
        parts.append(pd.DataFrame(np.clip(vals, lo, hi), index=grp.index, columns=grp.columns))
    X = pd.concat(parts).reindex(index=df.index, columns=cols)

    X = scale_features(X).replace([np.inf, -np.inf], np.nan)
    X.loc[:, :] = KNNImputer(keep_empty_features=True).fit_transform(X.values)
    X.loc[:, :] = np.clip(RobustScaler().fit_transform(X.values),
                          -config.ROBUST_CLIP, config.ROBUST_CLIP)
    return X


X = pipeline_features(df, cols_to_keep)
print(X.describe().T[['mean', 'std', 'min', 'max']])

# %% which features drive the site's umap X?
Y = df['umap X'].to_numpy()
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X, Y)

plt.barh(cols_to_keep, reg.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature importance for predicting umap X')
plt.pause(0.1)
plt.figure()

# %% site umap coloured by the first PC
pca = PCA(n_components=1, whiten=True)
plt.scatter(df['umap X'], df['umap Y'], c=pca.fit_transform(X)[:, 0], s=4)
plt.xlabel('umap X')
plt.ylabel('umap Y')
plt.title('Site UMAP, coloured by PC1')
plt.pause(0.1)

# %% grid plot with each feature on the umap
# Colour by the same log transforms the UMAP pipeline uses (config.SCALE_FEATURES),
# and clip the colour range to the 1-99th percentile so a few extreme cells
# don't squash the rest of the colormap.
X_plot = scale_features(df[cols_to_keep].copy())

n_features = len(cols_to_keep)
n_cols = 4
n_rows = int(np.ceil(n_features / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axs = axs.flatten()

for i, ax in enumerate(axs):
    if i >= n_features:
        ax.axis('off')
        continue
    col = X_plot[cols_to_keep[i]].to_numpy()
    vmin, vmax = np.nanpercentile(col, [1, 99])
    sc = ax.scatter(df['umap X'], df['umap Y'], c=col, cmap='viridis', vmin=vmin, vmax=vmax, s=4)
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel('umap X')
    ax.set_ylabel('umap Y')
    title = cols_to_keep[i]
    if cols_to_keep[i] in config.SCALE_FEATURES:
        title += f' ({config.SCALE_FEATURES[cols_to_keep[i]]})'
    ax.set_title(title)

plt.tight_layout()
plt.show()

# %% try other umap settings on the pipeline features
# Left: coloured by the site's umap X, to see how much structure carries over.
# Right: coloured by dandiset; islands made of a single dandiset are batch
# effects rather than cell types.
umap_model = UMAP(n_neighbors=150, metric='euclidean')
umap_embedding = umap_model.fit_transform(X)

codes = df['dandiset label'].astype('category').cat.codes
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=df['umap X'], s=4)
axs[0].set_title('coloured by site umap X')
axs[1].scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=codes, cmap='tab20', s=4)
axs[1].set_title('coloured by dandiset')
for ax in axs:
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
plt.tight_layout()
plt.show()

# %%
#patchOTDA
import ot.da as da
#pick a ref? 00020
Xt = X[df["dandiset label"] == "000020"]
dataset_adapted = {}
dataset_adapted_norm = X.copy()
for code in df["dandiset label"].unique():
    if code == "000020":
        continue
    Xs = X[df["dandiset label"] == code]

    dataset_adapted[code] = np.array(Xs.values).copy()

reg_grid = np.logspace(-1, 0.1, 5)




for reg in reg_grid:
    temp_adapted = {}
    da_ = da.JCPOTTransport(reg_e=reg, max_iter=100, log=True,  verbose=True)
    adapted = da_.fit_transform(Xs=list(dataset_adapted.values()), ys=[-1*np.ones(len(Xs)) for Xs in dataset_adapted.values()], Xt=np.array(Xt.values), yt=-1*np.ones(len(Xt)))

    #embedding after adaptation
    for i, code in enumerate(dataset_adapted.keys()):
        temp_adapted[code] = adapted[i]

    for i, code in enumerate(temp_adapted.keys()):
        dataset_adapted_norm.loc[df["dandiset label"] == code, :] = temp_adapted[code]

    reducer5 = UMAP(densmap=False, metric="euclidean",
                            n_neighbors=100, verbose=True)
    embedding = reducer5.fit_transform(dataset_adapted_norm)



    plt.scatter(embedding[:, 0], embedding[:, 1], s=3, c=df["brain_subregion"].astype("category").cat.codes)
    plt.title(f'UMAP embedding with reg_e={reg}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
   
    plt.show()


# %%
#try laplace

temp_adapted = {}
dataset_adapted_laplace = {}
for code in df["dandiset label"].unique():
    if code == "000020":
        continue
    print(f"Processing dandiset label: {code}")
    Xs = X[df["dandiset label"] == code]
    dataset_adapted_laplace[code] = np.array(Xs.values).copy()

    n_neighbors = min(len(Xs), 3)

    if n_neighbors < 3:
        print(f"Skipping dandiset label {code} due to insufficient neighbors")
        temp_adapted[code] = np.array(Xs.values).copy()
        continue

    da_ = da.EMDLaplaceTransport(reg_lap=10, max_iter=100, log=True, verbose=True, similarity_param=n_neighbors)
    adapted = da_.fit_transform(Xs=np.array(Xs.values), Xt=np.array(Xt.values))

    temp_adapted[code] = adapted

for i, code in enumerate(temp_adapted.keys()):
    dataset_adapted_norm.loc[df["dandiset label"] == code, :] = temp_adapted[code]

reducer6 = UMAP(densmap=False, metric="euclidean",
                        n_neighbors=100, verbose=True)
embedding = reducer6.fit_transform(dataset_adapted_norm)

plt.scatter(embedding[:, 0], embedding[:, 1], s=3, c=df["brain_subregion"].astype("category").cat.codes)
plt.title(f'UMAP embedding with Laplace adaptation')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
# %%
