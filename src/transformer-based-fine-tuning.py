# Author  : Oguzhan Ozcelik
# Date    : 14.07.2022
# Subject : Fine-tuning of Transformer-based language models used in our paper. 
# Models  : BERT, RoBERTa, mBERT, XLM-R, DistilBERTurk, BERTurk-32k, BERTurk-128k, ELECTRA-tr, ConvBERTurk

import time
import pandas as pd
from seqeval.metrics import precision_score, f1_score, recall_score, classification_report
from simpletransformers.ner import NERModel, NERArgs
import torch
import os
import csv
from transformers import logging

logging.set_verbosity_error()

news = ["data/news/", "news_results"]
wikiann = ["data/wikiann/", "wikiann_results"]
fbner = ["data/fbner/", "fbner_results"]
twner = ["data/twner/", "twner_results"]
atisner = ["data/atisner/", "atisner_results"]

paths = [news]  # add list names to train: wikiann, fbner, twner, atisner

# Transformer-based language models used in our paper
models = [
    ["auto", "dbmdz/bert-base-turkish-cased", "berturk32k"],
    ["auto", "dbmdz/bert-base-turkish-128k-cased", "berturk128k"],
    ["auto", "dbmdz/distilbert-base-turkish-cased", "distilberturk"],
    ["auto", "dbmdz/convbert-base-turkish-cased", "convberturk"],
    ["auto", "dbmdz/electra-base-turkish-cased-discriminator", "electra-tr"],
    ["auto", "bert-base-multilingual-cased", "mbert"],
    ["auto", "xlm-roberta-base", "xlm-r"],
    ["auto", "bert-base-cased", "bert-base-cased"],
    ["auto", "roberta-base", "roberta-base"],
]


args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 5e-5
args.overwrite_output_dir = True
args.train_batch_size = 16
args.max_seq_length = 128
args.no_save = True
REPEAT = 10

for path in paths:
    label = []

    if not os.path.isdir(path[1]):
        os.mkdir(path[1])

    if path[1] == "fbner_results":
        label = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
                 "B-MONEY", "I-MONEY", "B-PERCENT", "I-PERCENT", "O"]

    elif path[1] == "twner_results":
        label = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
                 "B-MONEY", "I-MONEY", "B-PERCENT", "I-PERCENT", "O"]

    elif path[1] == "atisner_results":
        label = ["B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
                 "B-CODE", "I-CODE", "B-MONEY", "I-MONEY", "O"]
    else:
        label = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    for mod in models:
        precision = []
        recall = []
        f_scores = []

        if not os.path.isdir(path[1] + '/' + mod[2]):
            os.mkdir(path[1] + '/' + mod[2])

        for i in range(0, REPEAT):
            model = NERModel(mod[0], mod[1], labels=label, args=args)  # initialize the model

            model.train_model(train_data=path[0]+'train_CONLL.txt', eval_data=path[0]+'test_CONLL.txt')  # train the model

            result, model_outputs, preds_list, true_list = model.eval_model(path[0] + 'test_CONLL.txt')  # test the model

			# logging the scores
            prec = precision_score(true_list, preds_list, average='weighted')
            rec = recall_score(true_list, preds_list, average='weighted')
            f1 = f1_score(true_list, preds_list, average='weighted')

            precision.append(prec)
            recall.append(rec)
            f_scores.append(f1)
            report = classification_report(true_list, preds_list, digits=4)
			
			# logging classification report
            class_report = path[1] + '/' + mod[2] + "/" + "entities_repeat_" + str(i)
            with open(class_report, "w") as text_file:
                text_file.write(report)

			# logging the predicted labels
            pred = path[1] + '/' + mod[2] + "/" + "pred_repeat_" + str(i)
            with open(pred, 'w') as f:
                wr = csv.writer(f, delimiter=' ')
                wr.writerows(preds_list)

        f_scores = [num for num in f_scores]
        precision = [num for num in precision]
        recall = [num for num in recall]

		# logging the final results of 10 random initialization
        result_txt = path[1] + '/' + mod[2] + "/" + "FINAL_RESULTS_" + mod[2]

        with open(result_txt, 'w') as f:
            f.write("Repeat#\tPrecision\tRecall\tF1_Score\n")
            for i in range(len(f_scores)):
                text = str(i) + "\t" + str(precision[i]) + "\t" + str(recall[i]) + "\t" + str(f_scores[i]) + "\n"
                f.write(text)
            text = "\nAvg.\t" + str(sum(precision) / len(precision)) + "\t" +\
                   str(sum(recall) / len(recall)) + "\t" + str(sum(f_scores) / len(f_scores))
            f.write(text)
