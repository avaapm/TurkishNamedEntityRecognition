# Named entity recognition in Turkish: A comparative study with detailed error analysis

## Overview
This repository contains the official implementation of "Named entity recognition in Turkish: A comparative study with detailed error analysis" paper. Additionaly, detailed evaluation results supported by statistical tests are provided.

This study provides a comparative analysis on the performances of the state-of-the-art approaches for Turkish named entity recognition using existing datasets with varying domains. The study includes a detailed error analysis that examines both quantitative (entity types, varying entity lengths, and changing word orders) and qualitative (ambiguous entities and noisy texts) factors that can affect the model performance.

## Environment
- Python 3.8.11
- PyTorch 1.11.0
- Tensorflow 2.6.0

To install the environment using Conda:
```bash
$ conda env create -f requirements.yml
```

This command creates a Conda environment named `ner_tr`. The environment includes all necessary packages for the training of the models in the study. After installation of the environment, activate it using the command below:
```bash
$ conda activate ner_tr
```

## Running
### Train
To train the models in this study, run the command below.
```bash
$ python main.py [R_MODE] [D_PATH] [M_PATH] [M_NAME] -r
```

| Parameter Name  | Type | Definition  |
| :-------------- | :--- | :---------- |
| `[R_MODE]` | `str` | Run mode: 'train' or 'test'|
| `[D_PATH]` | `str` | Path of the data folder containing train.tsv and test.tsv files |
| `[M_PATH]`| `str`  | Path for the model (save model when R_MODE='train', load when R_MODE='test') |
| `[M_NAME]`| `str`  | The name of the model (berturk_crf, bilstm, etc.) |
| `-r`| `str`  | Path for the evaluation report (use only in test mode) |

Example command is below to train BERTurk-CRF model.
```bash
$ python main.py train '/src/data/atisner/' '/models/berturk_crf/' berturk_crf
```

### Test

To test the fine-tuned models, run the command below.

Example command is below to train BERTurk-CRF model.
```bash
$ python main.py test '/src/data/atisner/' '/models/berturk_crf/' berturk_crf -r '/results/berturk_crf/'
```

## Citation
If you make use of this code, please cite the following paper:
```bibtex
@misc{ozcelik_and_toraman2022ner,
      title={Named entity recognition in Turkish: A comparative study with detailed error analysis}, 
      author={Oguzhan Ozcelik and Cagri Toraman},
      year={2022},
      url={https://github.com/avaapm/TurkishNamedEntityRecognition/}
}

```
