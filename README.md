# FaMeSumm
Code for EMNLP 2023 paper "FaMeSumm: Investigating and Improving Faithfulness of Medical Summarization"

Navigation:
[Overview](#overview), 
[Datasets](#datasets),
[Models and Experiments](#models-and-experiments),
[Repository Structure](#repository-structure)


## Overview
We introduce **FaMeSumm**, a framework to improve **Fa**thfulness for **Me**dical **Summ**arization. FaMeSumm is a general-purpose framework applicable to various language models on many medical summarization tasks. It adopts two objectives that finetune pre-trained language models to explicitly model faithfulness and medical knowledge. The first one uses contrastive learning that adopts **much simpler heuristics (as straightforward as rule-based copying and manipulating source texts)** than other contrastive learning baselines. The second objective learns medical knowledge by **modeling medical terms and their contexts** in the loss function. FaMeSumm delivers **consistent improvements over mainstream language models** such as BART, T5, mT5, and PEGASUS, yielding state-of-the-art performances on metrics for faithfulness and general quality. Human evaluation by doctors also shows that **FaMeSumm generates more faithful outputs**. The figure below shows a diagram of FaMeSumm architecture with an example reference summary. The underlined part in the reference contains a medical term (“Vitamin K”) and its context (“do not contain”) that are modeled by FaMeSumm.
![example](assets/FaMeSumm_diagram.png)