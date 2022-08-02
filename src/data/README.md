# Data folder

This folder contains five folders named for each dataset.

```
data
├── news
├── wikiann
├── twner
├── fbner
└── atisner
```

## Dataset folders

Each dataset folder includes two files `train.tsv` and `test.tsv` employed during the experiments.

Since we are not the curators of the datasets used in this study (except for ATISNER), we are not able to share them without permission; therefore, we provided sample tsv files for the datasets (News Articles, WikiANN-tr, TWNER and FBNER) in order to show the dataset file formats for the experimental process. In order to obtain the datasets, you can ask to original dataset curators cited in the dataset folders.

## Dataset file format
| Sentence # | Word | Tag |
| ------------- | ------------- | ------------- |
| Sentence: 0 | Dallas | B-LOC | 
| Sentence: 0 | 'a  | O | 
| Sentence: 0 | gidiş | O |
| Sentence: 0 | dönüş | O | 
| Sentence: 0 | yolculuk  | O | 
| Sentence: 0 | yapmak | O |
| Sentence: 0 | istiyorum | O | 
| Sentence: 0 | 'a  | O | 
| Sentence: 0 | gidiş | O |
| Sentence: 1 | philadelphia|B-LOC|
| Sentence: 1 |	'ye	| O |
| ... | ... | ... |

> We provided ATISNER dataset. One can directly use ATISNER dataset folder in order to reproducibility of the implementations.
