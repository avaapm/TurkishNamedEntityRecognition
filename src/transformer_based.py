# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Fine-tuning and testing of BERT, RoBERTa, distilBERT, mBERT, XLM-R, CovBERT and ELECTRA models
# via Simple Transformers library.

import os
import json
import shutil

from src.utils import DatasetLoader
from transformers import logging
from simpletransformers.ner import NERModel, NERArgs


class TRM_MODELS:
    def __init__(self, data_path, model_path, model_name):
        self.model_map = {'bert': "bert-base-cased", 'roberta': "roberta-base",
                          'mbert': "bert-base-multilingual-cased",
                          'xlm': "xlm-roberta-base", 'dberturk': "dbmdz/distilbert-base-turkish-cased",
                          'berturk32': "dbmdz/bert-base-turkish-cased",
                          'berturk128': "dbmdz/bert-base-turkish-128k-cased",
                          'electra_tr': "dbmdz/electra-base-turkish-cased-discriminator",
                          'convberturk': "dbmdz/convbert-base-turkish-cased"}

        logging.set_verbosity_error()
        if model_name not in list(self.model_map.keys()):
            raise ValueError("Invalid feature type. Expected one of: %s" % list(self.model_map.keys()))

        with open('src/configs/trm_models_config.json') as f:
            config = json.load(f)

        # Load dataset
        self.data_path = data_path  # input data path either train or test
        self.model_path = model_path  # model will be saved to this path or loaded for test
        self.model_name = model_name

        # Set arguments
        self.args = NERArgs()  # initialize NER arguments
        self.args.learning_rate = config['learning_rate']  # float: learning rate
        self.args.max_seq_length = config['max_seq_len']  # int: sequence length
        self.args.num_train_epochs = config['num_train_epochs']  # int: epoch number
        self.args.train_batch_size = config['train_batch_size']  # int: batch size
        self.args.eval_batch_size = config['test_batch_size']  # int: batch size
        self.args.no_save = False  # bool: whether to save model
        self.args.overwrite_output_dir = True  # bool: whether to write output_dir
        self.args.save_model_every_epoch = False
        self.args.save_eval_checkpoints = False
        self.args.save_steps = -1
        self.args.no_cache = True
        self.args.classification_report = True
        self.args.output_dir = self.model_path

    def train(self):
        dataset = DatasetLoader(self.data_path + 'train_sentenced.tsv', None)
        train_df = dataset.transformer_loader(train=True)

        model = NERModel("auto", self.model_map[self.model_name], dataset.train_tags, args=self.args)

        model.train_model(train_data=train_df)
        shutil.rmtree('runs')

    def evaluate(self, result_path):
        dataset = DatasetLoader(None, self.data_path + 'test_sentenced.tsv')
        test_df = dataset.transformer_loader(train=False)

        model = NERModel("auto", self.model_path, args=self.model_path + 'model_args.json')

        model.eval_model(test_df, wandb_log=False, output_dir=result_path)
        with open(result_path + 'eval_results.txt') as file:
            lines = file.read().splitlines()

        with open(result_path + 'final_result', 'w') as f:
            for line in lines[:-5]:
                f.write(f"{line}\n")

        os.remove(result_path + 'eval_results.txt')
