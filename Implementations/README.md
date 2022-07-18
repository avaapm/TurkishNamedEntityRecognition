# Implementations in our paper

NOTE: The implementation codes will be more structered soon.

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
