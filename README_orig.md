# Detecting Syntactic Change Using a Neural Part-of-Speech Tagger

perubahan syntactic

Code for 
[Detecting Syntactic Change Using a Neural Part-of-Speech Tagger](https://arxiv.org/abs/1906.01661), 

Syntactic --> rule

which will appear at the `1st International Workshop` on `Computational Approaches` `to Historical Language Change` at ACL 2019.

## Abstract

- method `train a diachronic LSTM`
- `analyze the tagger's ability` `to implicitly` `learn temporal structure between years`, and 
- extent `this knowledge` `can be transferred` `to date` `new sentences` (update transfered generating text)

diakronis = berkaitan dengan cara di mana sesuatu, terutama bahasa, telah berkembang dan berevolusi melalui waktu.

*We train `a diachronic` (2 achronic) `long short-term memory` (LSTM) `part-of-speech tagger` on a `large corpus` `of American English` from the `19th`, `20th`, and `21st` centuries. 
-
We `analyze the tagger's ability` `to implicitly` `learn temporal structure between years`, and `the extent` `to which` `this knowledge` `can be transferred` `to date` `new sentences`. 

`The learned year embeddings` `show` `a strong linear` `correlation` `between their first principal component` `and time`. 

We show that `temporal information` `encoded` `in the model` `can be used` `to predict` `novel sentences`' `years of composition` `relatively well`. (prediksi kalimat dengan penggubahan berdasarkan tahun)

`Comparisons to a feedforward baseline` `suggest` `that the temporal change` `learned` `by the LSTM` `is` `syntactic` `rather than purely lexical`. 

lexical -> struktur aturan bahasa

Thus, `our results` suggest (memberi kesan) that our tagger is `implicitly learning to model` `syntactic change` in American English over the course of the 19th, 20th, and early 21st centuries.*

results --> learned model for syntactic change in American English over the course of the 19th, 20th, and early 21st centuries

<!-- TODO: Add link to paper, citations, etc. -->

## Dependencies

Our implementation uses the following Python dependencies:
* argparse
* collections
* gensim
* numpy
* os
* pickle
* random
* re
* sklearn
* matplotlib
* statsmodels
* sys
* tensorflow

All of these libraries can be installed with pip.

## Getting Started

Please contact the authors for data.

`Once you have the raw data downloaded`, 

	data_processing.py, 
the data processing file, must be run first. 

Please specify 
- `EMBED_PATH` (the `location` `of the word embeddings` -- `do not include the name of the embedding file`), 
- `CORPUS PATH` (the location of the text files -- `do not include the name of any text file`), 
- `SAVE_PATH` (the `location where you would like to save` `the output` `embedding matrix`, 
	- `X_word_array`, 
	- `X_year_array`, and 
	- `Y_array`), and 
- `LEX_PATH` (the location of the lexicon file -- include the lexicon filename).

After the data has been downloaded and is located correctly, the data processing file can be run from the terminal using the command:

```
python dataprocessing.py
```

`The actual code` `to train` `and evaluate` `the LSTM` (

	lstm.py ) 
must be run second. 
We must 
- specify `DATA_PATH` (the location of the processed embedding matrix, `X_word_array`, `X_year_array`, and `Y_year_array` -- do not include any of the filenames), 
- `LEX_PATH` (the location of the lexicon file -- include the lexicon filename), 
- `TRAIN_SAVE_PATH`/`TEST_SAVE_PATH` (the location where you would like to `save` `the train` `and test data`, respectively), 
- `MODEL_PATH` (the location where you would like to save all model information), and 
- `PLOTS_PATH` (the location where you would like to save all plots).

After the data is processed, you can train an LSTM model and test it using:

```
python lstm.py --cut
```

The `--cut` flag specifies that you `want to create` `train` `and test` `data sets`. 
Once you have run the command with this flag once, you can leave it out in the future to `use previously generated train and test data sets`. (train test split and save it )

In addition, once you have a trained model, you can rerun the evaluation code without retraining by running:

```
python lstm.py --notrain
```

Additional `argparse` options can be found in the LSTM.py file.

## License

We obtained the rights to use the Corpus of Historical American English (COHA) through our affiliation with Yale University. 

Thank you Kevin Merriman for helping us get access to this corpus!















