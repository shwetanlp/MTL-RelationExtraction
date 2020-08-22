from nltk.tokenize import word_tokenize
import os
import numpy as np
import matplotlib
import pickle
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils import class_weight
from collections import Counter


DATA_DIR = "../data"

def get_entity_position(sentence_list, dataset, ENTITY_NAME, DATASETS):
    if get_task_name(dataset, DATASETS) == 'ddi':
        sentence = ' '.join(sentence_list)
        sentence = sentence.replace('DRUGA','DRUG')
        sentence = sentence.replace('DRUGB','DRUG')
        sentence_list = sentence.split(' ')
    c_list = Counter(sentence_list)
    pos=[]
    for index, word in enumerate(sentence_list):
        if get_task_name(dataset, DATASETS) != 'i2b2':
            if word.isupper() and c_list[word]==2 and word in ENTITY_NAME[dataset]:
                pos.append(index)
        elif get_task_name(dataset, DATASETS) == 'i2b2':
            if word.isupper() and word in ENTITY_NAME[dataset]:
                pos.append(index)
    return pos






def get_class_weights(y_train_data, train_data):
    class_weight_dicts={}
    for task_id, y_train in y_train_data.items():
        y_train = y_train[:train_data[task_id]]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train),
                                                          y_train)
        class_weight_dict = dict(enumerate(class_weights))
        class_weight_dicts[task_id] = class_weight_dict
    return class_weight_dicts


def get_task_name(task_id, DATASETS):
  return DATASETS[task_id]

def _load_raw_data_from_file(filename, task_id, ENTITY_NAME, DATASETS):
  data = []
  count=0
  n=0
  with open(filename, encoding='utf-8') as f:
    for line in f:
      segments = line.strip().split('\t')
      if len(segments) == 2:
        labels = segments[1].strip()
        sentence = segments[0].strip()
        tokens = sentence.split(' ')
        sentence =' '.join(tokens)
        pos = get_entity_position(tokens, task_id, ENTITY_NAME, DATASETS)
        if len(pos)==2:
            data.append((sentence, labels, pos))
            count=count+1
        else:
            continue
  print(filename)
  print(count)
  return data, len(data)

def _load_raw_data(dataset_name, task_id, FOLD, ENTITY_NAME, DATASETS):
  train_file = os.path.join(DATA_DIR, dataset_name+FOLD +'.train')
  train_data, train_len = _load_raw_data_from_file(train_file, task_id, ENTITY_NAME, DATASETS)
  test_file = os.path.join(DATA_DIR, dataset_name+FOLD+'.test')
  test_data, test_len = _load_raw_data_from_file(test_file, task_id, ENTITY_NAME, DATASETS)
  return (train_data, test_data, train_len, test_len)

def load_raw_data(fold, DATASETS, ENTITY_NAME):
    datasets = []
    train_lens = []
    test_lens = []
    fold="-"+fold
    for task_id, dataset in enumerate(DATASETS):
        train_data, test_data, train_len, test_len = _load_raw_data(dataset, task_id, fold, ENTITY_NAME, DATASETS)
        datasets.append((train_data, test_data))
        train_lens.append(train_len)
        test_lens.append(test_len)
    return datasets, train_lens, test_lens


def map_vocab_2_tokens(id_list, id2vocab, should_concat=False, isSentence =False):
    tokens=[]

    for id in id_list:
        if isSentence and id == 0:
            continue
        tokens.append(id2vocab[id])
    if should_concat:
        return ' '.join(tokens)
    return tokens

def Average(lst):
    return sum(lst) / len(lst)


def plot_curve(actual, predicted_prob, task, fold, mode):
    actual = np.asarray(actual)
    predicted_prob = np.asarray(predicted_prob)
    # FPR, TPR, _ = roc_curve(actual, predicted_prob[:, 1], pos_label='True')
    precision, recall, thresholds = precision_recall_curve(actual, predicted_prob[:, 1])
    # roc_AUC = auc(FPR, TPR, reorder=True)
    pr_auc = auc(recall, precision, reorder=True)
    # plt.figure()
    # plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % roc_AUC)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.02])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.savefig(task+'-roc.png')
    # to_save= (FPR, TPR)
    # with open(task+'-roc.pickle', 'wb') as handle:
    #     pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(task + '-' + mode +  '-'+ fold + '-pr.png')
    to_save = (precision, recall)
    with open(task + '-' + mode  + '-pr.pickle', 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pr_auc
