# Implementations in our paper

- `data` folder contains the dataset folders; however, we are not able to share the datasets without the permission of the original dataset curators.
- `fasttext` folder contains the pretrained fasttext word embeddings in order to use in machine learning and deep learning models. You can download the word embeddings under `fasttext` folder.

To run SVM and CRF:

```bash
$ python machine-learning-training.py
```

To run BiLSTM, BiGRU, CNN, BiLSTM-CRF, BiGRU-CRF:

```bash
$ python deep-learning-training.py
```

To run Transformer-based language models:

```bash
$ python transformer-based-fine-tuning.py
```

To run BERT-CRF:

```bash
$ python bert-crf-trainer.py
```

To run BERT-BiLSTM-CRF:

```bash
$ python bert-bilstm-crf-trainer.py
```

The results will be logged in folders named as 'dataset_name/model_name/'; for example, 'news_results/bert_crf/'.

> The implementation codes will be more structered soon. For now, please indicate the model name in the source codes following the directions.
