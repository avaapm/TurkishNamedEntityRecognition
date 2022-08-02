# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject:


import argparse
import sys
from src.svm import SVM_MODELS  # svm_hc, svm_ft
from src.crf import CRF_MODELS  # crf_hc, crf_ft
from src.dl_models import DL_MODELS  # BiLSTM, BiGRU, CNN
from src.transformer_based import TRM_MODELS  # BERT, RoBERTa, mBERT, XLM-R, distilBERTurk, BERTurk32k, BERTurk128k,
# ConvBERTurk, ELECTRA-tr
from src.hyb_dl_models import LSTM_GRU_CRF  # BiLSTM-CRF, BiGRU-CRF
from src.bert_crf import BERT_CRF_MODEL
from src.bert_bilstm_crf import BERT_BiLSTM_CRF_MODEL

models = ["svm_hc", "svm_ft", "crf_hc", "crf_ft", "bilstm", "bigru", "cnn", "bert",
          "roberta", "mbert", "xlm", "dberturk", "berturk32", "berturk128",
          "electra_tr", "convberturk", "bilstm_crf", "bigru_crf", "berturk_crf",
          "berturk_bilstm_crf"]


def main(run_mode, data_path, model_path, model_name, result_path):

    if run_mode == 'train':
        if model_name == 'svm_hc':
            running = SVM_MODELS(data_path=data_path, model_path=model_path, feature='hc')
            running.train()

        elif model_name == 'svm_ft':
            running = SVM_MODELS(data_path=data_path, model_path=model_path, feature='ft')
            running.train()

        elif model_name == 'crf_hc':
            running = CRF_MODELS(data_path=data_path, model_path=model_path, feature='hc')
            running.train()

        elif model_name == 'crf_ft':
            running = CRF_MODELS(data_path=data_path, model_path=model_path, feature='ft')
            running.train()

        elif model_name == 'bilstm':
            running = DL_MODELS(data_path=data_path, model_path=model_path, model_name='bilstm')
            running.train()

        elif model_name == 'bigru':
            running = DL_MODELS(data_path=data_path, model_path=model_path, model_name='bilstm')
            running.train()

        elif model_name == 'cnn':
            running = DL_MODELS(data_path=data_path, model_path=model_path, model_name='bilstm')
            running.train()

        elif model_name == 'bert':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='bert')
            running.train()

        elif model_name == 'roberta':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='roberta')
            running.train()

        elif model_name == 'mbert':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='mbert')
            running.train()

        elif model_name == 'xlm':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='xlm')
            running.train()

        elif model_name == 'dberturk':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='dberturk')
            running.train()

        elif model_name == 'berturk32':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='berturk32')
            running.train()

        elif model_name == 'berturk128':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='berturk128')
            running.train()

        elif model_name == 'electra_tr':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='electra_tr')
            running.train()

        elif model_name == 'convberturk':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='convberturk')
            running.train()

        elif model_name == 'bilstm_crf':
            running = LSTM_GRU_CRF(data_path=data_path, model_path=model_path, model_name='bilstm_crf')
            running.train()

        elif model_name == 'bigru_crf':
            running = LSTM_GRU_CRF(data_path=data_path, model_path=model_path, model_name='bilstm_crf')
            running.train()

        elif model_name == 'berturk_crf':
            running = BERT_CRF_MODEL(data_path=data_path, model_path=model_path)
            running.train()

        elif model_name == 'berturk_bilstm_crf':
            running = BERT_BiLSTM_CRF_MODEL(data_path=data_path, model_path=model_path)
            running.train()

    elif run_mode == 'test':

        if model_name == 'svm_hc':
            running = SVM_MODELS(data_path=data_path, model_path=model_path, feature='hc')
            running.evaluate(result_path=result_path)

        elif model_name == 'svm_ft':
            running = SVM_MODELS(data_path=data_path, model_path=model_path, feature='ft')
            running.evaluate(result_path=result_path)

        elif model_name == 'crf_hc':
            running = CRF_MODELS(data_path=data_path, model_path=model_path, feature='hc')
            running.evaluate(result_path=result_path)

        elif model_name == 'crf_ft':
            running = CRF_MODELS(data_path=data_path, model_path=model_path, feature='ft')
            running.evaluate(result_path=result_path)

        elif model_name == 'bilstm':
            running = DL_MODELS(data_path=data_path, model_path=model_path, model_name='bilstm')
            running.evaluate(result_path=result_path)

        elif model_name == 'bigru':
            running = DL_MODELS(data_path=data_path, model_path=model_path, model_name='bilstm')
            running.evaluate(result_path=result_path)

        elif model_name == 'cnn':
            running = DL_MODELS(data_path=data_path, model_path=model_path, model_name='bilstm')
            running.evaluate(result_path=result_path)

        elif model_name == 'bert':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='bert')
            running.evaluate(result_path=result_path)

        elif model_name == 'roberta':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='roberta')
            running.evaluate(result_path=result_path)

        elif model_name == 'mbert':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='mbert')
            running.evaluate(result_path=result_path)

        elif model_name == 'xlm':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='xlm')
            running.evaluate(result_path=result_path)

        elif model_name == 'dberturk':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='dberturk')
            running.evaluate(result_path=result_path)

        elif model_name == 'berturk32':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='berturk32')
            running.evaluate(result_path=result_path)

        elif model_name == 'berturk128':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='berturk128')
            running.evaluate(result_path=result_path)

        elif model_name == 'electra_tr':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='electra_tr')
            running.evaluate(result_path=result_path)

        elif model_name == 'convberturk':
            running = TRM_MODELS(data_path=data_path, model_path=model_path, model_name='convberturk')
            running.evaluate(result_path=result_path)

        elif model_name == 'bilstm_crf':
            running = LSTM_GRU_CRF(data_path=data_path, model_path=model_path, model_name='bilstm_crf')
            running.evaluate(result_path=result_path)

        elif model_name == 'bigru_crf':
            running = LSTM_GRU_CRF(data_path=data_path, model_path=model_path, model_name='bilstm_crf')
            running.evaluate(result_path=result_path)

        elif model_name == 'berturk_crf':
            running = BERT_CRF_MODEL(data_path=data_path, model_path=model_path)
            running.evaluate(result_path=result_path)

        elif model_name == 'berturk_bilstm_crf':
            running = BERT_BiLSTM_CRF_MODEL(data_path=data_path, model_path=model_path)
            running.evaluate(result_path=result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", help="Specify the running mode of the model_name as train or test",
                        type=str, choices=["train", "test"])
    parser.add_argument("data_path", help="The folder location of the input data",
                        type=str)
    parser.add_argument("model_path",
                        help="The model path. Used as the destination in training and input path in testing",
                        type=str)
    parser.add_argument("model_name", help="Select the model to run for classification",
                        type=str, choices=["svm_hc", "svm_ft", "crf_hc", "crf_ft", "bilstm", "bigru", "cnn", "bert",
                                           "roberta", "mbert", "xlm", "dberturk", "berturk32", "berturk128",
                                           "electra_tr", "convberturk", "bilstm_crf", "bigru_crf", "berturk_crf",
                                           "berturk_bilstm_crf"])
    parser.add_argument("-r", "--result_path", help="The location to store the result values for testing",
                        type=str, required=sys.argv[1] == "test")
    args = parser.parse_args()
    main(args.run_mode, args.data_path, args.model_path, args.model_name, args.result_path)
