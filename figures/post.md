I created a "map" of patch-clamp electrophysiology recordings across ~9 brain regions and 6 species (mouse, human, and four non-human primates)! 
I thought it would be a fun exercise to integrate all the open intracellular electrophysiology datasets from the (DANDI archive: https://dandiarchive.org/) into a single dataset, similar to projects in the transcriptomics world like (HYPOMAP: https://www.nature.com/articles/s42255-022-00657-y) and (MapMyCells: https://brain-map.org/bkp/analyze/mapmycells). Here, I gathered ~15,000 whole-cell patch-clamp recordings from 52 datasets (dandisets). Then I extracted 11 electrophysiological features from each cell (AP width, input resistance, sag, f-I slope, etc.) and embedded them within one over-arching dataset.

Attached is the UMAP embedding of the dataset. Here you can see separation by species, but also by region. I tried a rough clustering of the data, and we are able to tease out some interesting clusters, including a group of fast spikers and a group with large voltage "sag".

However, this comes with a few caveats. Most regions (and some species) only have one dandiset representing them (i.e. a single lab). Recording conditions (temperature, sampling rate, step protocol) also differ between labs, and these can alter the recorded features quite a bit. So some of this separation is likely a batch effect between labs, rather than a real biological difference. I go into this a bit more in the write-up: 

Included with this project is a browsable front end web database: [https://smestern.com/intraDANDI_explorer/](https://smestern.com/intraDANDI_explorer/). NOTE: much of this front end was built before the advent of LLM-assisted coding, and my front end skills aren't the best, so it's a bit wonky. I figured I would leave it as is for its charm!

Code: https://github.com/smestern/intraDANDI_explorer

Thanks to all the labs who have shared their data on DANDI! If you know of a dataset I missed, or have thoughts on how to tease apart lab effects from biology, let me know.
