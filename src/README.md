# Source codes of the models

## Source Folders
- `data` folder contains the dataset folders; however, we are not able to share the datasets (except for ATISNER) without the permission of the original dataset curators. It is explained how we accessed these datasets under each dataset folder.
- `fasttext` folder contains the pretrained fasttext word embeddings in order to use in machine learning and deep learning models. You can download the word embeddings under `fasttext` folder.
- `configs` folder contains the hyperparameters of the models used during the running.

## Implementation files
- `utils.py` file used for data preprocessing.
- `svm.py` file used for Support Vector Machine (SVM) model.
- `crf.py` file used for Conditional Random Fields (CRF) model.
- `dl_models.py` file used for deep learning models (BiLSTM, BiGRU and CNN).
- `transformer_based.py` file used for Transformer-based language models (BERT, RoBERTa, distilBERT, mBERT, XLM-R, ConvBERT and ELECTRA)
- `hyb_dl_models.py` file used for hybrid deep learning models (BiLSTM-CRF and BiGRU-CRF).
- `bert_crf.py` file used for BERT-CRF model.
- `bert_bilstm_crf.py` file used for BERT-BiLSTM-CRF model.

> The details of the model architectures are investigated in the paper.