# Toward Hybrid Code Representation Learning Models between Static and Contextualized Embeddings for Bug Detection

This repository contains the code for **Toward Hybrid Code Representation Learning Models between Static and Contextualized Embeddings for Bug Detection** and the [Page](https://vectorevaluationresearch.github.io/vectorevaluationresearch/) that has visualized data.

## Contents

1. [Website](#Website)
2. [Introduction](#Introduction)
3. [Dataset](#Dataset)
4. [Requirement](#Requirement)
5. [Instruction](#Instruction)
6. [Demo](#Demo)

## Website

We published the visualized experiment of our reasearch in https://vectorevaluationresearch.github.io/vectorevaluationresearch/

## Introduction

Polysemy in natural language texts is the coexistence of many
possible meanings for a word or phrase in different contexts. However,
due to different nature, in source code, some code tokens and
statements exhibit polysemy while other tokens (e.g., keywords,
separators, and operators) or statements maintain the same meaning
in different contexts. Polysemy is crucial in automated bug
detection because some tokens/statements with the same lexemes
are buggy in certain contexts, and not in others. Thus, several existing
bug detection approaches have moved from using static word
embedding models for code representation learning (CRL) to the
contextualized models (i.e., the embeddings are sensitive to the surrounding
contexts). While the intention and design mechanisms of
contextualized CRL models are to capture the contextual information
of program units in different representations (code sequences,
ASTs, graphs), the key questions on the use of static, contextualized,
or hybrid models remain unanswered due to the nature of mixed
polysemy in source code as explained.
We conducted our experiments on a total of 12 popular sequence-
/tree-/graph-based embedding models and on samples of a dataset of
10,222 Java projects with +14M methods. We modified and adapted
four evaluation metrics for contextuality from texts to code structures
to evaluate the vector representations produced by those CRL
models.We reported several important findings. For example, different
CRL models exhibit different levels of contextuality: the models
with higher contextuality help a bug detection model perform better
than the static ones. However, the most important finding is
that a simple hybrid model actually helps a machine-learning bug
detection model perform better than both static and contextualized
models. Thus, we call for actions to develop hybrid CRL models
that fit with the mixed polysemy nature of source code.

## Dataset

### Preprocessed Dataset

We published our processed dataset at https://drive.google.com/file/d/1WRxvz0Xj-IlTZN5N9qoXMLKKSZup4w41/view?usp=sharing

Please create a ```processed``` folder under the root folder of this repository, download the dataset, unzip it and put all files in ```./processed``` folder.

### Use your own dataset

If you want to use your own dataset, please prepare the data as follow:

1. There is one data file called ```data.npy```

2. ```data.npy``` includes a list of methods. For each bug, there are four types of information

	1> ```Bug_label```: A ```1*2``` matrix that represent if the method is buggy
	
	2> ```Token_label```: A ```N*1``` matrix that represent if the token should be processed by bert. ```N``` is the total token in the method (padded to the max one). 
	
	3> ```Bert_embedding```: A ```N*E``` matrix stores all bert embeddings. ```E``` is the embedding size.
	
	4> ```Word2vec_embedding```: A ```N*E``` matrix stores all word2vec embeddings. 
	
	```data.npy``` stores these info with a list as ```[[Bug_label, Token_label, Bert_embedding, Word2vec_embedding], ... ]```

## Requirement

Please check all required packages in the [requirement.txt]()

## Instruction

Change the ```split``` value to define your data splitting position.

Run ```main.py``` to see the result for the experiment. 

## Demo

For the testing purpose of running, please download our demo that contains a small set of data. Demo download: https://drive.google.com/file/d/1nLea0vpvG4KAyBGy9VTQwXbB53QmH8oP/view?usp=sharing

Put ```processed``` in the same folder of this repo and then run ```main.py``` to see the results.
