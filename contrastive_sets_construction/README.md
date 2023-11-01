# Constructing Contrastive Sets
Due to differences on dataset and language, we design positive and negative sets for each dataset with slightly different heuristics.

## Contrastive Sets for HQS
Run all blocks of `Contrastive_sets_construction_HQS.ipynb` to get the contrastive sets of HQS.


## Contrastive Sets for RRS
Follow these two steps:

    python Contrastive_sets_construction_RRS.py 

    python ad-hoc_cleanup_RRS.py 

For performance reason, we use `hoc_cleanup_RRS.py` to clean up the newly generated positive sets from `Contrastive_sets_construction_RRS.py`.

## Contrastive Sets for MDS
Run all blocks of `Contrastive_sets_construction_MDS.ipynb` to get the contrastive sets of MDS. Note that you need `ALL_medical_term_file_Chinese.txt` to build the contrastive set. It is a file that contains all the medical terms from training, validation, and test splits of MDS (the collection of `1_medical` and `2_medical`). We share it here to give everyone an idea how it looks like.
