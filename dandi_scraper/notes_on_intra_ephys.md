## A Map of Intracellular Electrophysiology Data on DANDI

I thought it would be a fun exercise to integrate all the open intracellular electrophysiology datasets from the [DANDI archive](https://dandiarchive.org/) into a single dataset, similar to projects in the transcriptomics world like [HYPOMAP](https://www.nature.com/articles/s42255-022-00657-y) and [MapMyCells](https://brain-map.org/bkp/analyze/mapmycells).

### TOC

* [1. What is This?](#what-is-this)  
* [2. Examine The Embeddings](#examining-the-embeddings)  
* [3. Is what we are seeing biological differences or a confound?](#is-what-we-are-seeing-biological-differences-or-a-confound-i-am-not-sure)  
    * [3.1. Pulling apart a few technical factors](#pulling-apart-a-few-technical-factors)  
    * [3.2. Can we tell the datasets apart?](#can-we-tell-the-datasets-apart)  
* [4. Interesting landmarks](#interesting-landmarks)  
    * [4.1. The clusters](#the-clusters)  
    * [4.2. The Fast Spikers and Broad Spikers](#the-fast-spikers-and-broad-spikers)  
    * [4.3. The sag islands](#the-sag-islands)  
    * [4.4. The latency corner](#the-latency-corner)  
* [5. The process](#the-process)  
    * [5.1. Getting a consistent protocol](#part-1-getting-a-consistent-protocol)  
    * [5.2. Getting the files through ipfx](#part-2-getting-the-files-through-ipfx)  
    * [5.3. Finding some metadata](#part-3-finding-some-metadata)  
    * [5.4. Building the map](#part-4-building-the-map)  


### What is this?

In this project I extracted features from whole cell patch clamp recordings. Whole-cell patch clamp is a technique where a small pipette is placed next to a single neuron's membrane, then the experimenter applies suction and breaks through the membrane. This makes the pipette continuous with the inside of the cell, allowing experimenters to 'listen' to the inner electrical workings of neurons. From here, neuroscientists can answer many questions about how neurons operate. For example, they might use 'voltage-clamp' (where the neuron's voltage is held at a specified level and membrane currents are recorded) to examine ionic currents. Or they might use 'current-clamp' (where current is applied via the patch pipette and neuron voltage is recorded) to examine how neurons react to changes in current.

In this project, I'm looking at "long pulse" ([see below](#part-1-getting-a-consistent-protocol)) current clamp recordings from cells across the nervous system. The protocol includes both subthreshold and suprathreshold components, allowing researchers to obtain a general profile of how the neuron behaves, similar to how transcriptomics research takes a "snapshot" of a cell's transcriptomic phenotype. Here we use the long pulse to get a "snapshot" of the neuron's electrophysiology, where we can observe things like its spikes, rebounds, and subthreshold oscillations, giving us an electrophysiological fingerprint (like the ephys phenotyping from [Gouwens et al. 2020](https://pubmed.ncbi.nlm.nih.gov/33186530/)). Let's look at some examples:

Here you can see the uniqueness of cell firing patterns. Some cells, like cell 4, show an irregular phenotype, while others, like cell 0, show a regular spiking phenotype. From here, we'll extract features from the traces using the Allen Institute's [Intrinsic Physiology Feature Extractor (IPFX)](https://ipfx.readthedocs.io/en/latest/). We'll extract both spiking and subthreshold features to get a comprehensive "fingerprint" of the neuron. Below we've embedded these features for all ~15,000 cells in the dataset.

## Examining the Embeddings
Lets take a look at the overall embedding of the data.
![alt text](<../figures/blog/2x/Artboard 1@2x.png>)  
Fig 1: UMAP embedding of all 15,869 cells from 52 dandisets. Top left: cells coloured by species. Bottom left: cells coloured by brain region (neocortex and unknown are greyed out so the smaller regions stand out). Right: the six coarse clusters, with three example cells from each cluster (circles mark where each example sits on the map).

As a sanity check, we can look at how features map over a UMAP. Notably, there is a lot of controversy about [interpreting feature distributions on UMAPs](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288). However, we are just using it as a general sanity check to ensure the UMAP is embedding logically and not exploiting any bias in the data.

Here, we can see the UMAP is clustering feature values together in a somewhat coherent manner. Promisingly (to me) is the mapping -x to +x mapping of action potential width, a common feature for very rudimentary electrophysiological cell typing in the cortical regions (https://www.nature.com/articles/s41593-019-0417-0). 

From here, we can also look at cells' K-NN neighbours in the dataset, a sanity checks to see if the cells match qualitatively across datasets. We would expect cells with similar electrophysiology phenotypes to group together

![alt text](../figures/blog/cluster_examples_grid_preview.png)

The results are interesting. Qualitatively, cells with similar ephys fingerprints do seem to be neighbours (regardless of species). 

# Is what we are seeing biological differences or a confound? (I am not sure)

Above I presented the idea that the embeddings do show some coherence in feature grouping. However, one may note that both species and region remain separated in the embedding. This presents an interesting potential biological difference that we could investigate. However, this observation comes with a few caveats:
- Most regions only have one dandiset to represent them (i.e. a single origin lab)
- Most species also have a single origin lab (save for the Allen Inst.), or also differ in region from the remaining data
- Recording conditions differ between many dandisets: temperature, sampling rate, and even the long-pulse protocol itself (step duration and step size)

Given that ephys lab protocols (solutions, temperature, time of slicing, etc.) can drastically alter recorded ephys features, and that these protocols vary slightly (or drastically) between labs, it becomes tricky to separate real biological differences from 'batch effects' between labs.

First we can look at how the raw features differ between dandisets, regions, and species:  
![](../figures/umap_workup_tf20/06_dandiset_feature_medians.png)  
Fig 2: Median value of each feature per dandiset, scaled so that 0 is the population median (red = higher, blue = lower). Label colour is the lab the dandiset came from.

Here we can see that features are generally skewed between datasets. For example, dandiset "000148" features recordings from spinal cord interneurons in the mouse. We can see the median input resistance (Rin) and time constant (tau) are quite a bit higher than the other dandisets. This is consistent with reported high values in mouse spinal cord interneurons: [https://neuroelectro.org/neuron/175.html](https://neuroelectro.org/neuron/175.html). So it's possible that this is a real biological signal; however, without an alternate dandiset in the spinal cord, we cannot fully make that conclusion.  
Another example is the high "sag" (see below) reported in the pigtail macaque. This is consistent with reported values: https://www.cell.com/cell/fulltext/S0092-8674(26)01278-X (in fact I believe these datasets are part of this publication). These datasets originate from the Allen Inst., so comparing between Allen Inst. datasets could be considered valid, in which case this is likely a real biological observation.  

The best check is to look at how much of each feature's variance can be explained by species, then region, then dandiset:
![03_variance_partition.png](notes_on_intra_ephys_files/03_variance_partition.png)  
Fig 3: How much of each feature's variance is explained by where the cell came from. Dark blue is the share explained by species alone, light blue is the extra share explained by adding region (within species), and orange is the extra share explained by adding dandiset (within species x region). The bars stack, so the full length of each bar is the share explained by dandiset alone. Everything left over (the gap to 1.0) is variation between cells within a dandiset. Computed as a simple between-group R² on the same scaled features fed into the UMAP.

A few things stand out to me here. For every feature, at least half of the variance is *not* explained by any of these labels. It sits between cells within the same dandiset, which is what we would hope for if the map is mostly capturing cell types.

The features with the largest dandiset share (beyond species and region) are f-I slope (~0.23), AP peak (~0.22), AP width (~0.18) and AP fast trough (~0.14). These are mostly spike shape features, which are exactly the ones we would expect to be sensitive to recording temperature and acquisition settings. The f-I slope I suspect is partially a protocol thing, as the datasets use different step sizes (from 2 pA up to 50 pA) and step durations (0.5 to 1 s). On the other hand, the passive features (Rin, tau, V baseline) and first spike latency are mostly explained by species and region, with only a small extra dandiset term. Sag has the largest species term of any feature (~0.17), which lines up with the macaque sag story above.

There are two big caveats to reading this plot, though:
- Since most regions (and many species) only come from a single dandiset, the blue bars are not clean 'biology' either. If a region only has one dataset, any lab effect gets credited to the region first. So I would read the blue bars as an upper bound on the biology, and the orange bars as a lower bound on the lab effect.
- A 'dandiset effect' is not necessarily technical. Labs target different cells. For example, dandiset 000020 is enriched for GABAergic cells (~25% fast spikers) whereas 001455 has essentially none. Fast spikers peak ~14-23 mV below the broad spikers in every lab, so some of the orange in AP peak and width is just which cells each lab chose to patch.

### Pulling apart a few technical factors

To get a better feel for the orange bars, I checked a few of the technical suspects directly:

- Temperature: The Tolias lab has two mouse neocortex dandisets, 000008 (room temp) and 000035 (recorded warmer). Same lab, same species, same region, which makes for a nice comparison. The median AP width drops from ~1.16 ms to ~0.68 ms (roughly 40% narrower) and the AP peak drops by ~10 mV in the warmer set, while tau and threshold barely move. This fits with the temperature effect on [ion channel kinetics](https://link.springer.com/article/10.1007/BF00584655) mentioned below.
- Sampling rate: Most of the Allen data is sampled at 50 kHz, whereas many of the other labs (including the spinal cord sets) sampled at 10 kHz. To test this, I took Allen cells and degraded them down to 10 kHz, then re-ran the feature extraction. This only explained ~10% of the AP width gap between the 10 and 50 kHz datasets, and almost none of the AP peak gap. AP upstroke is the most sensitive (about a third of the gap, more with heavier filtering). The passive features did not change at all. So sampling rate is a factor for upstroke, but it does not explain most of the differences.
- Protocol: For about half the cells, the rheobase sweep contains a single spike, so the 'firing rate at rheobase' is really just 1/(step duration). Coarser current steps also overshoot rheobase. Because of this I swapped rheobase firing rate out for the f-I slope.

![](../figures/downsampling_test/02_effect_vs_real_gap.png)  
Fig 4: The effect of degrading the same 50 kHz Allen cells (blue bars) compared to the real difference between the 10 kHz and 50 kHz datasets (orange line). For width and peak, downsampling barely closes the gap.

### Can we tell the datasets apart?

Another way to ask the question is: if I hand a classifier cells from two dandisets, can it tell which is which? (0.5 = indistinguishable, 1.0 = perfectly separable.)

![](../figures/umap_workup_tf20/05_pair_auc_by_kind.png)  
Fig 5: Pairwise classifier accuracy (AUC) between every pair of dandisets, split by how similar the pair is on paper.

Dandisets from the same species and region but a different lab are almost perfectly separable (median AUC 0.99), just as separable as dandisets from different species. Dandisets from the same lab, species and region are still distinguishable (median 0.83), but much less so. To me, this suggests that the lab is a major factor. However, the many Allen human datasets mix together well in the UMAP, so when the protocol is standardized, datasets do overlap. Dandiset 001538 (mouse striatum) forms its own island. I think this one is real biology: it is the only striatal dataset, and the cells look like textbook spiny projection neurons (very hyperpolarized baseline, low Rin, almost no sag, long latency to first spike).

I also tried a whole bunch of UMAP settings and feature subsets to see if I could get the datasets to mix better. Unfortunately I really couldn't find one with a good balance. The only thing that removes the dataset structure is centring every dandiset on itself, but that also wipes out the real differences (e.g. the striatal island and the primate sag). I also attempted my PatchOTDA technique, this did work well for integration, but I worried it presented the same problem. So, for now, I chose not to correct for anything.

So is it biology or a confound? I think it is likely a bit of both. The big features (fast vs broad spikers, sag, latency) seem to hold up across labs. But I would be very cautious about reading species or region differences off the map when they come from a single lab. As more labs deposit data covering the same regions, we should be able to tease these apart.

# Interesting landmarks

## The clusters

To get a rough overview of the map, I split the cells into clusters. Here, I used a Gaussian mixture model (GMM) on the same scaled features used to build the UMAP (not on the UMAP coordinates themselves). The clustering is done in two tiers: first the cells are split into 6 coarse clusters, then each coarse cluster is split into up to 5 fine clusters (30 in total). I'll mostly stick to the 6 coarse clusters here (Fig 1, right). The number of clusters is a setting I picked, so I would not read too much into the exact boundaries.

![](../figures/blog/cluster_feature_heatmap_preview.png)  
Fig 6: Median value of each feature per coarse cluster. The number in each box is the median, and the colour shows how far that median sits from the population median (in IQRs; red = higher, blue = lower).

A rough tour of the clusters:
- Cluster 1 - "Regular spiking-ish" (n=5,921): The largest cluster. Broad, tall spikes (~0.95 ms wide, ~44 mV peak), a lower input resistance and a shallow f-I slope. This is roughly what I would expect from pyramidal (excitatory) neurons, and only ~1% of its cells are fast spikers. It also picks up most of the amygdala (75%) and striatum (80%) cells, and most of the Tolias lab cells (74%).
- Cluster 2 - "Regular spiking interneurons" (I think) (n=4,112): Narrower spikes (~0.66 ms), a higher input resistance (~220 MΩ) and a steeper f-I slope than cluster 1. Over half of this cluster comes from dandiset 000020, the Allen Inst. mouse visual cortex Patch-seq set, which focuses on GABAergic cells. So I suspect this cluster is mostly non-fast-spiking interneurons.
- Cluster 3 - "Longer latency" (n=2,198): This cluster is defined almost entirely by its long latency to the first spike (median ~570 ms, vs ~60-140 ms for the other clusters). This is the latency corner discussed below. About half of these cells come from dandisets with an "Unknown" region (mostly Allen Inst. sets), and it also takes about half of the brainstem cells.
- Cluster 4 - "Sag" (n=1,890): High sag (median ratio of 0.34, vs ~0.04-0.09 for the other clusters) and a slightly more depolarized baseline. This is the primate cluster: 45% human and ~38% non-human primate, vs only 17% mouse. This lines up with sag having the largest species term in Fig 3.
- Cluster 5 - "Fast burst" (n=994): Small, slow spikes (upstroke ~120 V/s, about half that of the other clusters, ~17 mV peak, ~1.4 ms wide), along with a high input resistance and slow tau. This cluster collects most of the non-cortical cells: 92% of the spinal cord cells, 64% of the basal forebrain cells, and about half of the pallidum cells.
- Cluster 6 - "Fast spikers" (n=754): Narrow spikes (~0.42 ms), a fast tau (~7 ms), low input resistance, and a very steep f-I slope (~0.8 Hz/pA). 87% of these cells pass the usual fast spiking cutoff (narrow width and a low upstroke:downstroke ratio). These are likely Pv-like interneurons.

![](../figures/blog/region_by_cluster_preview.png)  
Fig 7: The share of each brain region's cells that falls in each cluster.

Of course, the caveats from above apply here too, and a couple of the clusters show it pretty clearly:
- Cluster 6 is 94% one dandiset (000020). There are plenty of fast spikers elsewhere in the data, but they mostly don't end up here. Of the ~700 primate fast spikers, only 22 land in cluster 6; most of them (~390) land in the sag cluster (4) instead. Even about a third of 000020's own fast spikers land in cluster 2. The primate fast spikers have a lot more sag than the mouse ones (median sag ratio of 0.30 vs 0.05). Since 93% of these primate cells also come from the Allen Inst. (the same lab as 000020), I lean towards this being a real species difference. However, I have not looked into it further.
- Cluster 5 is enriched for the 10 kHz datasets. A quarter of its cells were sampled at 10 kHz, compared to ~5% of the whole dataset. Since AP upstroke is the feature most affected by sampling rate (see above), some of this cluster is likely technical. That said, the 50 kHz cells in this cluster also have slow upstrokes (median ~130 V/s), so it is not only a sampling effect.

So, I would treat the clusters as a handy way to navigate the map, rather than as cell types. With that in mind, let's look at a few landmarks in more detail.

## The Fast Spikers and Broad Spikers:

As noted above, spike width is a common metric for rapid, rough cell typing of cortical neurons (particularly in the extracellular electrophysiology world). The fast spikers are generally thought to be putative inhibitory neurons (likely Pv cells), whereas the broad spikers are thought to be putative excitatory pyramidal neurons. We now know that this distinction is not as clear-cut, as not all inhibitory neurons show "fast spikes" (see [Gouwens et al. 2019](https://www.nature.com/articles/s41593-019-0417-0)), and non-cortical regions do not have such clear cell-types (save for maybe the hippocampus). Regardless, as a first pass, looking at the distinction between the narrow and broad spikers seems like a good idea. The UMAP shows a pretty clear separation of spike width along the x-axis

The rightmost plot shows the first action potential from each cell. Here, you can see a pretty clear distinction between the fast spiker (cell 296) and the broad spiker (cell 1315). With 296 showing almost double the width. The caveat is that these cells are from disparate dandisets. Cell 296 is from dandiset 000020 (https://dandiarchive.org/dandiset/000020), whereas cell 1315 comes from 000008 (https://dandiarchive.org/dandiset/000008?search=000008&pos=1). These dandisets differ not only in their region (V1 vs motor), but also in their recording temperature (000020 at physiological, 000008 at room temp). Recording temp can have a large effect on [ion channel kinetics](https://link.springer.com/article/10.1007/BF00584655) and therefore action potential width.

## The sag islands:

There are two distinct islands in the UMAP embedding with a large measured "Sag". Sag in this case refers to the deflection present in the voltage when a cell is stimulated with a hyperpolarizing pulse. 

![image.png](notes_on_intra_ephys_files/image.png)

Image: A example of an HCN driven 'sag' from: https://doi.org/10.1016/j.neuropharm.2014.02.012


You may see a cell with sag described as having a h-current (or Ih), and this h-current is [thought to be driven by HCN channels](https://www.cell.com/neuron/fulltext/S0896-6273(18)30900-0). H-current bestows a number of different computation properites on neurons and expression differs across species and age (check out: [here](https://www.cell.com/neuron/fulltext/S0896-6273(18)30900-0), [or here](https://academic.oup.com/cercor/article/31/2/845/5927870?login=true)). Here sag is expressed as a ratio between steady state and absolutle deflection, you may see papers report absoulte or relative deflection in mV instead.

Interestingly, we identified two cells with different sag phenotypes, both from the human dataset. 

## The latency corner:

The feature 'latency_0_long_square' refers to the latency to the first spike. A cell with a long latency may have distinct computational properties that regulate its overall excitability. We can see that cells with a long latency cluster into the northeast corner of the UMAP space.

Here are two cells with high latency values. The waveforms of the cells show a characteristic 'shoulder' or 'hinge' that is indicative of an A-type current. In my primary brain region of study (the paraventricular nucleus [PVN] of the hypothalamus), we see a lot of [A-type current](https://doi.org/10.1111/j.1469-7793.2000.t01-1-00193) in the neuroendocrine magnocellular neurons. Critically, this a-type current is an important regulator of cellular excitability, with the a-type currents being a target of [NMDA-driven modulation in magnocellular neurons](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/jphysiol.2014.270793). Its really cool to see it here too!

# The process

***NOTE**: Most of this code was originally written before the various updates to the NWB standard and the package `ipfx`. Many of the problems discussed below are now fixed through various QoL updates. Since then, the extraction code has also moved into my patch-clamp package `gigaseal` (`gigaseal.database.protocol_detect` and `gigaseal.database.long_square`), and the snippets below are the current versions.

## Part 1: Getting a consistent protocol

The first step in collating the data is finding some common features to analyze. With this in mind, I had to choose a common protocol for extracting features. In this case, I chose to extract features from the "long-pulse" current-clamp stimulation protocol. A "long-pulse" protocol is a semi-standard protocol used by patch-clampers to characterize fundamental intrinsic properties of neurons. All of the above voltage traces come from long-pulse protocols. In most cases, the protocol consists of a baseline state, and then a 'long' (typically 0.7 - 2s) epoch of current stimulation. For each sweep of stimulation, the applied current is incremented, typically ranging from -100 to 100+ pA. 


![image.png](notes_on_intra_ephys_files/image.png)  
IMAGE: An example long pulse protocol from the [allen cell types of database](https://celltypes.brain-map.org/). 

However, I ran into a problem. Many of the datasets used did not have an explicitly labelled 'long-pulse'. As such, we had to figure out which sweeps in each dataset were the 'long-pulse' ones. I tried several methods to easily detect long-pulse sweeps, including machine learning techniques. In the end, I relied on simple conditional programming to filter out the long pulse. 

My first version treated any change in the command current as a step edge. This broke on datasets where the command carries line noise, since every sample becomes an 'edge'. The current version finds the step edges against a robust estimate of the noise floor, then checks that the step is flat (not a ramp) and that the command returns to its pre-step baseline (allowing for a holding current, rather than requiring the command to start and end at exactly 0). The code looks like this:

Two more problems came up as I added more dandisets. First, many files contain more than one long-pulse run (for example, the hyperpolarizing and depolarizing steps run as separate protocols, or a repeat of the protocol later in the recording). Now, the sweeps that pass are grouped into runs by their step duration, and one run is picked per cell. The ranking prefers a step duration closest to 1 s, then a run with both hyperpolarizing and depolarizing steps (that actually reaches spiking voltages), then the earliest run in the recording (most likely to be at baseline), and finally the run with the most sweeps.

Second, some datasets store the whole protocol as a single long sweep. For example, dandiset 001454 records each cell as one ~70 s sweep with a 0.5 s step every 2 s. These are now detected and split into one sweep per step before the matching above.

## Part 2: Getting the files through ipfx

The next step was processing all the files using a consistent method. For the electrophysiological feature extraction, I relied on Allen Institute's excellent [ipfx](https://ipfx.readthedocs.io/en/latest/). When I started this project, ipfx was still in its infancy. Ipfx supported NWBs but had a few clauses specific to Allen Institutes-generated NWBs that made it hard to use with third-party-generated NWBs. Regardless, we will be using ipfx's [long-square analysis](https://ipfx.readthedocs.io/en/latest/auto_examples/analysis_examples/lsq_analysis.html#sphx-glr-auto-examples-analysis-examples-lsq-analysis-py) to extract consistent features for each cell.

In the end, I had to override a few ipfx functions to get third-party NWB's to pass through. For example, the old `get_stim_characteristics` function used by ipfx to identify long pulses assumes the presence of a short test-pulse (250ms) at the start of each sweep. Unfortunately, if a dataset did not feature a test pulse, it would fail at this step. My first override just skipped the first step if it fell within the first 250 ms. However, test pulses can also come *after* the main stimulus (e.g. dandiset 001261 has a trailing -20 pA test pulse). Now, the override splits the command into excursions away from baseline and keeps the longest one as the stimulus. It also measures the amplitude relative to the holding current, rather than the raw command:

From here, we will now run an analysis script across all the files in the dataset. In this case, we will use a custom version of the `data_for_specimen_id` function from ipfx. The function gets the features in 4 steps:

1. Load the data. Here I use my own loader (from `gigaseal`), which also fixes units. Some files log voltage in volts with an inconsistent NWB conversion factor (sometimes a multiplier, sometimes a divisor), so the loader picks whichever scaling gives a plausible membrane potential.
2. Step through each sweep and guess which sweeps are long pulses (also run some rudimentary QC), then pick one run for the cell (see Part 1).
3. Run the feature extractor.
4. Collect the features for rheobase and the steps above rheobase, plus the f-I slope.

I also had to tune a few of ipfx's spike detection settings. I kept ipfx itself (rather than writing my own detector) so the features stay comparable to the Allen Inst. datasets:
- dv/dt filter: ipfx low-pass filters the voltage at 10 kHz before taking dv/dt, which is above Nyquist for cells sampled at 10 kHz. Those cells get capped at 4 kHz, so now every cell is filtered at 4 kHz to put them all on the same footing.
- Spike threshold: ipfx sets threshold at 5% of the mean upstroke by default. In fast trains the threshold would slide back to the AHP trough, so I raised this to 20%.
- Detection window: in some datasets (000008, 000035), the voltage leaves baseline slightly before the recorded current step, so on strong steps the first spike was never detected. The spike detector's window now opens 0.5 ms before the step.
- Steps above rheobase: each sweep is now assigned to the nearest 20 pA step above rheobase. Requiring an exact multiple of 20 pA left cells recorded with other step sizes (e.g. 10, 25, 100 pA) with only their rheobase sweep.

Failures are also recorded per cell (which stage failed and why), so I can trace every missing cell back to where it was dropped.

## Part 3: Finding some metadata

I wanted to pull both the species and the brain region for each recorded cell. NWB has several metadata fields; however, I found that the metadata was not consistently placed on the asset or dandiset page. To this end, I opted to try out my first LLM interface project. Here, I asked the LLM to extract the region and species. I prompted the LLM with the dandiset title, description, and asset metadata. 

This has since changed a bit. To make sure the labels are right, the brain region now comes from a per-dandiset table that I reviewed by hand (`dandiset_metadata.csv`). Each dandiset gets a coarse region (e.g. "Neocortex") plus a finer subregion where one is stated (e.g. "motor cortex"), along with the lab it came from. The LLM (Cohere) now only suggests a row when a new dandiset shows up. It has to pick from a fixed list of regions, and its row is flagged as unreviewed until I check it. Species now comes from each cell's NWB file (matched on the NCBI taxonomy ID where possible), since some dandisets mix species (e.g. 001261 has both mouse and macaque). A version of the code is below.

## One final check:

As a final sanity check, we can examine the feature distributions and perform a rough QC. Many of the features have known reasonable physiological ranges. For example, we know that v_baseline probably should be higher than -100 mV and below 0 mV (being generous). 

These bounds now live in `config.py` and are applied when the dandisets are merged: a cell is dropped if any of its features falls outside the range. They are generally very liberal, but a few of them catch real detection errors. For example, a threshold below -70 mV usually means the detector picked up the stimulus onset rather than a spike, and a first spike less than 1 ms after the step is a stimulus artifact.

## Part 4: Building the map

Finally, the dandisets are merged into one table and embedded:

1. Drop cells that fail the QC bounds above, cells missing more than half of the features used for the map, and duplicate cells.
2. Fill in the remaining missing values with a KNN imputer (within each dandiset), and clip each feature to its 0.5 - 99.5th percentile.
3. Log-scale the skewed features (Rin, tau, AP width and upstroke), then robust-scale everything and clip at ±5 IQRs, so no single feature can dominate.
4. Run UMAP on the result.

The map uses 11 features: input resistance, tau, baseline voltage, sag, the first spike's threshold, peak, upstroke, width and fast trough, the f-I slope, and the latency to first spike. I originally also used the mean AP features and the firing rate at rheobase, but dropped them. The mean AP features correlate almost perfectly with the first AP versions (so keeping both double-weights spike shape), and the firing rate at rheobase is mostly just 1/(step duration) (see above).
