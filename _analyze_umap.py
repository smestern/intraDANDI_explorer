#%% import the necessary libraries
import json
import numpy as np
import sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import umap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

#%% load the data.js
df = pd.read_csv('all_new.csv')

#get the columns 
cols_to_keep = {
    "ap_1_width_0_long_square": "Rheo-AP width Log[(ms)]",
    "sag_nearest_minus_100": "Sag",
    "input_resistance": "Input resistance (MOhm)",
    "tau": "Tau Log[(ms)]",
    "v_baseline": "Baseline voltage (mV)",
    }
cols_to_keep = list(cols_to_keep.keys())

cols_to_keep = ['input_resistance', 'tau', 'v_baseline', 'sag_nearest_minus_100',
       'sag_measured_at', 'rheobase_i', 'fi_linear_fit_slope',
       'ap_1_threshold_v_0_long_square', 'ap_1_peak_v_0_long_square',
       'ap_1_upstroke_0_long_square', 'ap_1_downstroke_0_long_square',
       'ap_1_upstroke_downstroke_ratio_0_long_square',
       'ap_1_width_0_long_square', 'ap_1_fast_trough_v_0_long_square',
       'ap_mean_threshold_v_0_long_square', 'ap_mean_peak_v_0_long_square',
       'ap_mean_upstroke_0_long_square', 'ap_mean_downstroke_0_long_square',
       'ap_mean_upstroke_downstroke_ratio_0_long_square',
       'ap_mean_width_0_long_square', 'ap_mean_fast_trough_v_0_long_square',
       'avg_rate_0_long_square', 'latency_0_long_square',
       'stimulus_amplitude_0_long_square']

# %%
#filter the data
df = df.dropna(subset=['umap X'])
X = df[cols_to_keep]
Y = np.copy(df['umap X'].to_numpy())
#threshold y as greater than 5


# %%
#scale the data
scaler = StandardScaler()
impute = SimpleImputer(strategy='mean')
X = impute.fit_transform(X)
X = scaler.fit_transform(X)

#run regression
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X, Y)

#run PCA on X
pca = PCA(n_components=1, whiten=True)
pca.fit(X)
# %%
#show feature importance
importances = reg.feature_importances_

# %%
#plot the feature importance
import matplotlib.pyplot as plt
plt.barh(cols_to_keep, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()
# %%
#plot umap of the data
plt.scatter(df['umap X'], df['umap Y'], c=pca.fit_transform(X)[:,0])
#plt.scatter(reg.predict(X), df['umap Y'])
plt.xlabel('umap X')
plt.ylabel('umap Y')

plt.show()
#%% plot pca
#plt.scatter(pca.fit_transform(X)[:,0], pca.fit_transform(X)[:,1], c=df['umap X'])

#%% grid plot with each feature on the umap
fig, axs = plt.subplots(5, 5, figsize=(20,20))
for i, ax in enumerate(axs.flat):
    ax.scatter(df['umap X'], df['umap Y'], c=X[:,i])
    ax.set_xlabel('umap X')
    ax.set_ylabel('umap Y')
    ax.set_title(cols_to_keep[i])


# %%
from sklearn import tree

#plot the tree
plt.figure(figsize=(20,20))
tree.plot_tree(reg.estimators_[0], feature_names=cols_to_keep, filled=True)
plt.show()
# %%
