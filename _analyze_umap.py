# %% import the necessary libraries
from sklearn import tree
import matplotlib.pyplot as plt
import json
import numpy as np
import sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import umap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# %% load the data.js
df = pd.read_csv('all_new.csv')

# get the columns
cols_to_keep = {
    "ap_1_width_0_long_square": "Rheo-AP width Log[(ms)]",
    "sag_nearest_minus_100": "Sag",
    "input_resistance": "Input resistance (MOhm)",
    "tau": "Tau Log[(ms)]",
    "v_baseline": "Baseline voltage (mV)",
}
cols_to_keep = list(cols_to_keep.keys())

cols_to_keep = ['input_resistance', 'tau', 'v_baseline', 'sag_nearest_minus_100', 'rheobase_i',
                'ap_1_threshold_v_0_long_square', 'ap_1_peak_v_0_long_square',
                'ap_1_upstroke_0_long_square', 'ap_1_downstroke_0_long_square',
                'ap_1_upstroke_downstroke_ratio_0_long_square',
                'ap_1_width_0_long_square', 'ap_1_fast_trough_v_0_long_square',
                'ap_mean_threshold_v_0_long_square', 'ap_mean_peak_v_0_long_square',
                'ap_mean_upstroke_0_long_square', 'ap_mean_downstroke_0_long_square',
                'ap_mean_upstroke_downstroke_ratio_0_long_square',
                'ap_mean_width_0_long_square', 'ap_mean_fast_trough_v_0_long_square',
                'avg_rate_0_long_square', 'latency_0_long_square',]


# %%
# filter the data
df = df.dropna(subset=['umap X'])
X = df[cols_to_keep]
Y = np.copy(df['umap X'].to_numpy())
# threshold y as greater than 5

# %%
# Before scaling, let's check the distribution of the features
print(X.describe())

# %%
# scale the data
scaler = RobustScaler()
impute = SimpleImputer(strategy='mean')
X = impute.fit_transform(X)
X = scaler.fit_transform(X)

# run regression
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X, Y)

# run PCA on X
pca = PCA(n_components=1, whiten=True)
pca.fit(X)
# %%
# show feature importance
importances = reg.feature_importances_

# %%
# plot the feature importance
plt.barh(cols_to_keep, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.pause(0.1)
plt.figure()
# %%
# plot umap of the data
plt.scatter(df['umap X'], df['umap Y'], c=pca.fit_transform(X)[:, 0])
# plt.scatter(reg.predict(X), df['umap Y'])
plt.xlabel('umap X')
plt.ylabel('umap Y')

plt.pause(0.1)
# %% plot pca
# plt.scatter(pca.fit_transform(X)[:,0], pca.fit_transform(X)[:,1], c=df['umap X'])

# %% grid plot with each feature on the umap
# Colour by the same log transforms the UMAP pipeline uses (config.SCALE_FEATURES),
# and clip the colour range to the 1-99th percentile so a few extreme cells
# don't squash the rest of the colormap.
from dandi_scraper.analysis import scale_features
from dandi_scraper.config import SCALE_FEATURES
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
    if cols_to_keep[i] in SCALE_FEATURES:
        title += f' ({SCALE_FEATURES[cols_to_keep[i]]})'
    ax.set_title(title)

plt.tight_layout()
plt.show()
# %%

# # plot the tree
# plt.figure(figsize=(20, 20))
# tree.plot_tree(reg.estimators_[0], feature_names=cols_to_keep, filled=True)
# plt.show()
# # %%
