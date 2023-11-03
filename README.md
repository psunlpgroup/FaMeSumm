# FaMeSumm
Code for EMNLP 2023 paper "FaMeSumm: Investigating and Improving Faithfulness of Medical Summarization"

Navigation:
[Overview](#overview), 
[Datasets](#datasets),
[Contrastive Sets Construction](#contrastive-sets-construction),
[Models and Experiments](#models-and-experiments),
[Acknowledgement](#acknowledgement),
[Citation](#citation)


## Overview
We introduce **FaMeSumm**, a framework to improve **Fa**thfulness for **Me**dical **Summ**arization. FaMeSumm is a general-purpose framework applicable to various language models on many medical summarization tasks. It adopts two objectives that finetune pre-trained language models to explicitly model faithfulness and medical knowledge. The first one uses contrastive learning that adopts **much simpler heuristics (as straightforward as rule-based copying and manipulating source texts)** than other contrastive learning baselines. The second objective learns medical knowledge by **modeling medical terms and their contexts** in the loss function. FaMeSumm delivers **consistent improvements over mainstream language models** such as BART, T5, mT5, and PEGASUS, yielding state-of-the-art performances on metrics for faithfulness and general quality. Human evaluation by doctors also shows that **FaMeSumm generates more faithful outputs**. The figure below shows a diagram of FaMeSumm architecture with an example reference summary. The underlined part in the reference contains a medical term (“Vitamin K”) and its context (“do not contain”) that are modeled by FaMeSumm.

![example](assets/FaMeSumm_diagram.png)


## Datasets
We have tested FaMeSumm on different kinds of datassets to demonstrate its capability on various medical summarization tasks: Health Question Summarization (HQS),  Radiology Report Summarization (RRS), and Medical Dialogue Summarization (MDS). We are not allowed to share these datasets due to legal concerns, so we recommed to collect them by yourself. You may need to complete corresponding user agreement or crawl data on your own.
1. HQS: The first task of [MEDIQA 2021](https://sites.google.com/view/mediqa2021). The goal of this task is to summarize potentially complex consumer health questions.
2. RRS: The third task of [MEDIQA 2021](https://sites.google.com/view/mediqa2021). This task comes with two test splits (benchmarks): Indiana and Stanford. The goal of this task is to summarize the textual findings of radiology report written by radiologists.
3. MDS: A private Chinese dataset for medical dialogue summarization. We train models on this dataset to test their capabilities on understanding and summarizing doctor-patient conversations. To reproduce a dataset similar to our MDS, please refer to Appendix B of our paper to see the detailed data collection process.

Once the raw datasets are collected, check `sample_datasets` folder to see how we format data instance of each dataset. You need to match the formats in order to run experiments. Specifically, `question` represents a patient question that needs to be summarized, `summary` represents a reference summary, `1_medical` represents medical terms that appears in a reference summary only, `2_medical` represents medical terms that appears in both reference summary and source text/conversation/question, `neg_uni` represents the negative unigrams that exist in the reference, `content` contains a few `utterance` by different `speaker` of a medical dialogue, and `text` represents the source text of RRS dataset. In RRS, we concatenate the findings and background of the original dataset to form `text`. In MDS, `description` is usually the first utterance from the patient of a specific dialogue, and `type-B` is the reference summary.


## Contrastive Sets Construction
After datasets are built, please refer to [contrastive_sets_construction](/contrastive_sets_construction) folder to build constrastive sets. These sets are necessary for running the our training pipeline.

## Models and Experiments
First of all, install all required Python packages with `pip install -r requirements.txt`. We have tested these library versions to ensure the reproducibility of the reported performance scores when trained checkpoints are provided.

We provide the trained checkpoints of FaMeSumm models below:
* [FaMeSumm + PEGASUS on HQS](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/EedGT4rB3p9Oh-VhN0S05hMBfMaWRmeP_13JxDnalOcnhQ?e=MRMXMQ)
* [FaMeSumm + BART on HQS](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/ES-_bacefARDgDbqoLdgD9IBjNS0kZBBeGeNT33LPmbclg?e=6eTaak)
* [FaMeSumm + T5 on HQS](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/EUXXGGNNZ0ZDvKesNYNBSS8BrmVCSXZ1S3HKRsvEPVWZzw?e=fIVRsY)
* [FaMeSumm + BioBART on HQS](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/EUiqjFY7YJlGnmQvAoUzObUBVtS3tmwB-rNkYVTWtUlvog?e=dZDfzE)
* [FaMeSumm + PEGASUS on RRS-Indiana](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/EQ63sMuonjVPoYQflrdJYGwBBzD09E8xezPNKdtCCWstXA?e=m08Pv2)
* [FaMeSumm + PEGASUS on RRS-Stanford](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/Eap7sPuQVHdCpa9TyBCNS0AB-oyrnpbBRQbeR_EDUPDFUQ?e=ACpVrN)

Without using these checkpoints, you may fine-tune language models:

    # fine-tune PEGASUS on HQS
    python train_pega_HQS.py

    # fine-tune BART on HQS
    python train_bart_HQS.py

    # fine-tune T5 on HQS
    python train_t5_HQS.py

    #fine-tune BioBART on HQS
    python train_BioBART_HQS.py

    #fine-tune PEGASUS on RRS (code is the same for both Indiana and Stanford except that different validation sets are used)
    python train_pega_RRS.py

    #fine-tune mT5 on MDS
    python train_mt5_MDS.py

You will need to provide the API key of your WandB account in line 36 or 38. Please place the contrastive sets into a folder called `P&N` inside your dataset folder. All these training files require `ALL_medical_term_file_train.txt`, so you will need to collect all medical terms (the collection of `1_medical` and `2_medical`) of the training set for each dataset. You may check `ALL_medical_term_file_Chinese.txt` in [contrastive_sets_construction](/contrastive_sets_construction) as an example.

To do inference, refer to our code to run trained PEGASUS model below. Please change your checkpoint name at line 652. For other language models, you will need to replace the `PegaFineTuner` class in line 87 with the class you see in the corresponding fine-tuning file (e.g., `T5FineTuner`). You will also need to replace lines 585 to 595 with the corresponding lines in the fine-tuning file due to tokenizer differences.

    python test_pega_HQS.py

You will get ROUGE scores and C F1 after running the inference code. For other automatic metrics reported in our paper, we prepare [additional_metrics](/additional_metrics) folder and it contains the code for you to test model performance on different types of data. Note that each file in `additional_metrics` has its own requirements (e.g., trained model checkpoints and python environment), so please refer to its paper and/or GitHub repository to set things up.


## Acknowledgement
* Our fine-tuning code is developed based on: https://github.com/priya-dwivedi/Deep-Learning/blob/master/wikihow-fine-tuning-T5/Tune_T5_WikiHow-Github.ipynb.


## Citation
