# HECTOR

## Requirements
The code was tested with `python==3.7.3`.

The required libraries are listed in [requirements.txt](requirements.txt).

## Data

1. Download datasets from here:
    - [MAG-CS](https://drive.google.com/file/d/1P_MWGSy0JVq-nPpNQNtD6VbnF3gQfd3C/view?usp=sharing)
    - [PubMed](https://drive.google.com/file/d/1I-PnPhTF81G5lXi_GhEXe2gJJ-zawKqT/view?usp=sharing)
    - [EURLex](https://drive.google.com/file/d/1uBH-o_kZbKJxXhtFdWnzutuGHUGgcK9U/view?usp=sharing)
2. Unpack to `<DATA_DIR>`.
3. In `config/config_*.ini`, change `Paths.data_dir` value to `<DATA_DIR>`.

### Data format
#### Datasets:
- Each dataset contains the following files: `train.json`, `dev.json` and `test.json`.
- Each file contains N samples, one sample per line.
- Each sample is a dict with the following important keys:
  - text: original document.
  - title: original title.
  - **text_processed**: normalized title + document (lower case, no stopwords, no punctuation). 
  Used as *input to the model*.
  - **label**: list of relevant labels, where each label is a string. Model *target*.


- The rest of the keys is legacy from [original datasets](https://github.com/yuzhimanhua/MATCH) and can be useful for running other baselines.

#### Additional files:
Each dataset contains to additional files, required for training `ontology.json` and `taxonomy.txt`: 

**Ontology**: 

- `JSONL` file, each line describes a single label (key-value mapping):
  - label: string with label identifier.
  - title: label title in natural language.
  - definition: label definition in natural language.
  - txt: normalized title + definition.
  - level: level of the label in a the label tree.

**Taxonomy**:

- `TXT` file, each line contains space-separated labels, where first label is a parent and the rest are children. 


#### GloVe embeddings
The model is initialized with GloVe embeddings (840B tokens, 2.2M vocab, cased, 300d vectors). 
Please download from the [official website](https://nlp.stanford.edu/projects/glove/) and put next to the `<DATA_DIR>` (the exact path must be specified in `config["Paths"]["glove_model"]`).


## Training and Evaluation

Paths, training hyper-parameters and other model configurations depends on a dataset and are specified in corresponding config files: `config/config_*.ini`.

### Training

To run the training, use the following command:

```
python main.py --config CONFIG --name NAME
```

- `CONFIG`: path to a config file
- `NAME`: model name prefix

The code evaluates a validation loss after each epoch and save the best model to `./models/` directory.


Note that the model's code was designed to be trained using GPU acceleration and there is no CPU support.


### Prediction

The script implements beam search algorithm starting from given prefixes (label refinement task). 
The prefixes are constructed from labels of level < `LEVEL` assigned to a test instance.

To perform predictions using a trained model, use the following command:

```
python predict.py --config CONFIG --model MODEL --level LEVEL --output OUTPUT
```

- `CONFIG`: path to a config file
- `MODEL`: path to a trained model
- `LEVEL`: for label refinement task: level from which the prediction starts.
For example, when `LEVEL==2`, the model is provided with path prefixes of length 1 and start predicting labels from level 2.
For predicting from scratch (without prefixes), set `LEVEL` to 1.
- `OUTPUT`: path stub for output files. 

The script will generate two files: `<OUTPUT>-labels.npy` and `<OUTPUT>-scores.npy` with top-1000 predicted labels and their scores, respectively.

### Evaluation

The script evaluates model predictions generate by `predict.py` module. It can also be used for evaluation of other baseline methods which produce output in the same format 
([AttentionXML](https://github.com/yourh/AttentionXML), [MATCH](https://github.com/yuzhimanhua/MATCH)).

The scripts calculate the following metrics:
- `Precision@k` (`k` = 1, 3, 5)
- `NDCG@k` (`k` = 1, 3, 5)


To run the evaluation, use the following command:

```
python evaluation.py --testset TESTSET --pred PRED --ontology ONTOLOGY --level LEVEL
```

- `TESTET`: path to `test.json` file
- `PRED`: path to `<OUTPUT>-labels.npy` file (see above)
- `ONTOLOGY`: path to `ontology.json` file
- `LEVEL`: for label refinement task: only consider labels of level >= `LEVEL`. 
For all labels, set `LEVEL` to 1.

## References
For the full method description and experimental results please refer to our paper:

Natalia Ostapuk, Julien Audiffren, Ljiljana Dolamic, Alain Mermoud, and Philippe Cudre-Mauroux. 2024.
Follow the Path: Hierarchy-Aware Extreme Multi-Label Completion for Semantic Text Tagging.
*In Proceedings of the ACM Web Conference 2024 (WWW ’24), May 13–17, 2024, Singapore, Singapore.* 