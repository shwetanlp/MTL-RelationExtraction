import numpy as np
np.random.seed(1357) # for reproducibility
import os
import numpy as np
from collections import defaultdict
import re
import sys
import os
from statistics import mean
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Embedding
from keras.layers import Dense, RepeatVector, Dropout, TimeDistributed, Input, Lambda, Flatten, Concatenate, concatenate, Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, CuDNNGRU
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import backend as K
from time import time
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers, constraints
from keras.utils import np_utils
from keras.activations import softmax
from sklearn.model_selection import StratifiedKFold
import sklearn as sk
from sklearn.metrics import classification_report
import json
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score, auc, confusion_matrix
from utils import *
from flipGradientTF import *
import tensorflow as tf
import math
from functools import reduce
import argparse
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main(params):
    exp_result = {}
    for global_fold in range(params.TOTAL_FOLD):
        global_fold = global_fold+1
        global_fold = str(global_fold)
        result_to_store = {}
        print("###################  Fold-"+global_fold+"######### is being processed")

        class SelfAttention(Layer):
            def __init__(self, d_a_size=350, r_size=10, return_attention_vector=False, name='Self-Att', **kwargs):
                self.init = initializers.get('glorot_normal')
                self.name = name
                self.d_a_size = d_a_size
                self.r_size = r_size
                self.return_attention_vector = return_attention_vector
                super(SelfAttention, self).__init__(**kwargs)

            def build(self, input_shape):
                assert len(input_shape) == 3
                hidden_size = input_shape[-1]
                self.hidden_size = hidden_size
                self.W_s1 = self.add_weight(name='{}_W_s1'.format(self.name),
                                            shape=(hidden_size, self.d_a_size),
                                            initializer=self.init)
                self.W_s2 = self.add_weight(name='{}_W_s2'.format(self.name),
                                            shape=(self.d_a_size, self.r_size),
                                            initializer=self.init)
                self.trainable_weights = [self.W_s1, self.W_s2]
                super(SelfAttention, self).build(input_shape)  # be sure you call this somewhere!

            def call(self, x, mask=None):
                x_Ws1 = K.tanh(K.dot(x, self.W_s1))
                H = K.dot(x_Ws1, self.W_s2)
                A = softmax(H, axis=1)
                A_reshape = K.permute_dimensions(A, pattern=[0, 2, 1])
                M = K.batch_dot(A_reshape, x, axes=(2, 1))
                if self.return_attention_vector:
                    return [A_reshape, M]
                return M

            def get_output_shape_for(self, input_shape):
                if self.return_attention_vector:
                    return [(input_shape[0], self.r_size, input_shape[1]), (input_shape[0], self.r_size, input_shape[2])]
                return (input_shape[0], self.r_size, input_shape[2])

            compute_output_shape = get_output_shape_for


        def get_LayerOutput(model1, x_val, layer_name):
            for layer in model1.layers:
                if layer_name in str(layer.name):
                    intermediate_layer_model = Model(input=model1.input, output=model1.get_layer(layer.name).output)
                    intermediate_output = intermediate_layer_model.predict(x_val)
                    return intermediate_output
            return None

        def generate_data(datasets):
            texts = []
            train_labels = {}
            train_sentences = {}
            test_labels = {}
            test_sentences = {}
            train_pos = {}
            test_pos = {}
            for task_id, dataset in enumerate(datasets):
                label_list = []
                sentence_list = []
                pos_list = []
                train_d, test_d = dataset
                for tokens, label, pos in train_d:
                    texts.append(tokens)
                    label_list.append(label)
                    pos_list.append(pos)
                    sentence_list.append(tokens)
                train_labels[task_id] = label_list
                train_pos[task_id] = pos_list
                train_sentences[task_id] = sentence_list
                label_list = []
                sentence_list = []
                pos_list = []
                # train_d, test_d = dataset
                for tokens, label, pos in test_d:
                    texts.append(tokens)
                    label_list.append(label)
                    sentence_list.append(tokens)
                    pos_list.append(pos)
                test_labels[task_id] = label_list
                test_sentences[task_id] = sentence_list
                test_pos[task_id] = pos_list
            return texts, train_sentences, train_labels, test_sentences, test_labels, train_pos, test_pos


        def get_label_dict(dataset_name):
            if dataset_name == 'aimed' or dataset_name == 'bioinfer':
                label2id = dict([('False', 0), ('True', 1)])
            if dataset_name == 'ddi':
                label2id = {'false': 0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
            if dataset_name == 'i2b2':
                label2id = {'NONE': 0, 'TeCP': 1, 'TrCP': 2, 'TrAP': 3, 'PIP': 4, 'TeRP': 5}
            if len(label2id)>0:
                id2label = {v: k for k, v in label2id.items()}
                return id2label, label2id
            raise Exception


        def process_dataset(texts, train_sentences, train_labels, test_sentences, test_labels):
            tokenizer = Tokenizer(nb_words=params.MAX_NB_WORDS)
            tokenizer.fit_on_texts(texts)
            word2id = tokenizer.word_index
            train_cat_labels = {}
            train_padded_sentences = {}
            test_cat_labels = {}
            test_padded_sentences = {}
            id2labels_dic = {}
            for task_id, train_seq in train_sentences.items():
                sequences = tokenizer.texts_to_sequences(train_seq)
                train_padded_sentences[task_id] = pad_sequences(sequences, maxlen=params.MAX_SEQUENCE_LENGTH, padding='post',
                                                                truncating='post')

            for task_id, test_seq in test_sentences.items():
                sequences = tokenizer.texts_to_sequences(test_seq)
                test_padded_sentences[task_id] = pad_sequences(sequences, maxlen=params.MAX_SEQUENCE_LENGTH, padding='post',
                                                               truncating='post')

            for task_id, train_seq in train_labels.items():
                id2label, label2id = get_label_dict(get_task_name(task_id, params.DATASETS))
                id2labels_dic[task_id] = (id2label, label2id)
                labels = [label2id[label.strip()] for label in train_seq]
                labels = np.asarray(labels, dtype='int32')
                labels = np_utils.to_categorical(labels, num_classes=len(id2label))
                train_cat_labels[task_id] = labels

            for task_id, test_seq in test_labels.items():
                id2label, label2id = id2labels_dic[task_id]
                labels = [label2id[label.strip()] for label in test_seq]
                labels = np.asarray(labels, dtype='int32')
                labels = np_utils.to_categorical(labels, num_classes=len(id2label))
                test_cat_labels[task_id] = labels
            return word2id, id2labels_dic, train_padded_sentences, train_cat_labels, test_padded_sentences, test_cat_labels


        def generate_task_labels(train_labels, test_labels):
            train_task_labels = {}
            test_task_labels = {}
            for task_id, train_label in train_labels.items():
                size = len(list(train_label))
                task_label = [task_id] * size
                task_label = to_categorical(task_label, num_classes=params.N_TASK)
                train_task_labels[task_id] = task_label
            for task_id, test_label in test_labels.items():
                size = len(list(test_label))
                task_label = [task_id] * size
                task_label = to_categorical(task_label, num_classes=params.N_TASK)
                test_task_labels[task_id] = task_label
            return train_task_labels, test_task_labels




        def get_embedding_matrix(word_index):
            print('Indexing word vectors')
            word2vec = KeyedVectors.load_word2vec_format(params.EMBEDDING_FILE, binary=True)
            print('Found %s word vectors of pretrained embeddings' % len(word2vec.vocab))
            embedding_matrix = np.random.random((len(word_index) + 1, params.EMBEDDING_DIM))
            for word, i in word_index.items():
                if word in word2vec.vocab:
                    embedding_matrix[i] = word2vec.word_vec(word)
            print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
            return embedding_matrix


        def compute_adv_loss(s_feature, true_label, disc_model):
            s_flip = Flip(s_feature)
            task_label = disc_model(s_flip)
            adv_loss = K.mean(K.categorical_crossentropy(true_label, task_label))
            return adv_loss


        datasets, train_length, test_length = load_raw_data(global_fold, params.DATASETS, params.ENTITY_NAME)
        max_sample_length = max(train_length)
        # with open('processed_data.pkl', 'wb') as handle:
        #     pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # datasets= pickle.load(open('processed_data.pkl','rb'))
        texts, train_sentences, train_labels, test_sentences, test_labels, _, _ = generate_data(
            datasets=datasets)
        word2id, id2labels_dic, train_padded_sentences, train_cat_labels, test_padded_sentences, test_cat_labels = \
            process_dataset(texts, train_sentences, train_labels, test_sentences, test_labels)
        train_task_labels, test_task_labels = generate_task_labels(train_labels, test_labels)
        id2word = {value:key for key, value in word2id.items()}
        embedding_matrix = get_embedding_matrix(word2id)
        best_valid_predictions = {}
        embedding_layer = Embedding(len(word2id) + 1,
                                    params.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=params.MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def private_model(task_index):
            input = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = embedding_layer(input)
            embedding =Dropout(params.d_rate)(embedding)
            lstm_output = embedding
            for layer in range(params.N_GRU):
                lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE, return_sequences=True))(lstm_output)
            A, lstm_output = SelfAttention(r_size=params.r_size, return_attention_vector=True)(lstm_output)
            lstm_output = Flatten()(lstm_output)
            hidden_output = Dense(params.MLP_SIZE, activation='relu')(lstm_output)
            final_feature = hidden_output
            final_feature = Dropout(params.d_rate, name='private_feature')(final_feature)
            preds = Dense(params.NO_OF_LABELS[task_index], activation='softmax')(final_feature)
            model = Model(inputs=input, outputs=preds)
            model.compile(optimizer='adam', loss='categorical_crossentropy'
                      , metrics=['accuracy'])
            print("Private-feature-model")
            print(model.summary())
            return model

        def shared_model():
            input = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = embedding_layer(input)
            embedding = Dropout(params.d_rate)(embedding)
            lstm_output = embedding

            if params.isSelfAttention:
                for layer in range(params.N_GRU):
                    lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE, return_sequences=True))(lstm_output)
                A, lstm_output = SelfAttention(r_size=params.r_size, return_attention_vector=True)(lstm_output)
                lstm_output = Flatten()(lstm_output)
            else:
                for layer in range(params.N_GRU-1):
                    lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE, return_sequences=True))(lstm_output)
                lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE))(lstm_output)

            hidden_output = Dense(params.MLP_SIZE, activation='relu')(lstm_output)
            model = Model(inputs=input, outputs=hidden_output)
            return model



        def discriminator():
            s_feature = Input(shape=(params.MLP_SIZE,), dtype='float32')
            disc = Dense(params.N_TASK, activation='softmax')
            task_label = disc(s_feature)
            disc_model = Model(inputs=s_feature, outputs=task_label)
            disc_model.compile(optimizer='nadam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
            print(disc_model.summary())
            return disc_model

        def basic_model(task_index, disc_model=None):
            input = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32')
            task_label = Input(shape=(params.N_TASK,))
            embedding = embedding_layer(input)
            embedding =Dropout(params.d_rate)(embedding)
            lstm_output = embedding

            if params.isSelfAttention:
                for layer in range(params.N_GRU):
                    lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE, return_sequences=True))(lstm_output)
                A, lstm_output = SelfAttention(r_size=params.r_size, return_attention_vector=True)(lstm_output)
                lstm_output = Flatten()(lstm_output)
            else:
                for layer in range(params.N_GRU-1):
                    lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE, return_sequences=True))(lstm_output)
                lstm_output = Bidirectional(CuDNNGRU(params.HIDDEN_SIZE))(lstm_output)

            hidden_output = Dense(params.MLP_SIZE, activation='relu')(lstm_output)
            s_feature = s_model(input)
            p_feature = hidden_output
            final_feature = concatenate([s_feature, p_feature], -1)
            final_feature = Dropout(params.d_rate)(final_feature)
            preds = Dense(params.NO_OF_LABELS[task_index], activation='softmax')(final_feature)
            model = Model(inputs=[input, task_label], outputs=preds)

            if params.isAdv:
                adv_loss = compute_adv_loss(s_feature, task_label, disc_model)
                adv_loss = adv_loss * 0.05
                model.add_loss(adv_loss)
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.metrics_tensors.append(adv_loss)
                model.metrics_names.append("adv_loss")

            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print(model.summary())
            return model





        def evalaute(task_id_c, models, is_visualize_attention=False):

            for task_id in range(params.N_TASK):
                if task_id!=task_id_c:
                    continue
                print(get_task_name(task_id, params.DATASETS))
                preds = models[task_id].predict(
                        [test_padded_sentences[task_id], test_task_labels[task_id]])

                preds_prob = preds
                preds = np.argmax(preds, axis=1)
                labels = np.argmax(test_cat_labels[task_id], axis=1)
                id2label = id2labels_dic[task_id][0]

                result_to_store_local = {}
                result_to_store_local[task_id] = {}
                result_to_store_local[task_id]['confusion_matrix'] = confusion_matrix(labels, preds)
                result_to_store_local[task_id]['classification_report'] = classification_report(labels, preds, digits=4)

                prec_list, rec_list, f1_list = [], [], []


                prec, reca, fscore, sup = precision_recall_fscore_support(labels, preds, average='macro')
                prec_list.append(prec)
                rec_list.append(reca)
                f1_list.append(fscore)
                prec, reca, fscore, sup = precision_recall_fscore_support(labels, preds, average='micro')
                prec_list.append(prec)
                rec_list.append(reca)
                f1_list.append(fscore)
                prec, reca, fscore, sup = precision_recall_fscore_support(labels, preds, average='weighted')
                prec_list.append(prec)
                rec_list.append(reca)
                f1_list.append(fscore)

                result_to_store_local[task_id]['precision'] = prec_list
                result_to_store_local[task_id]['recall'] = rec_list
                result_to_store_local[task_id]['f-score'] = f1_list
                return result_to_store_local, prec_list, rec_list, f1_list, preds_prob, labels

        def train_disc(epochs, private_model, batch_size=params.batch_size):
            g_batch_num = [0] * params.N_TASK
            n_batch = int(max_sample_length / batch_size)
            disc_model = discriminator()
            private_features_train={}
            private_features_test = {}
            print("Generating private features...")
            for task_id in range(params.N_TASK):
                x_train, y_train = train_padded_sentences[task_id], train_cat_labels[task_id]
                task_train = train_task_labels[task_id]

                private_features_train[task_id] = get_LayerOutput(private_model[task_id], x_train,
                                        'private_feature')
                task_test = test_task_labels[task_id]
                x_test = test_padded_sentences[task_id]
                private_features_test[task_id] = get_LayerOutput(private_model[task_id], x_test,
                                            'private_feature')
            print('done...')
            for epoch in range(epochs):
                disc_losses,  disc_accuracies = [], []
                print("Epoch started ....", epoch)
                prev_index = 0
                for batch in range(n_batch):
                    index = (batch + 1) * batch_size
                    for task_id in range(params.N_TASK):
                        if index <= train_length[task_id]:
                            g_batch_num[task_id] = g_batch_num[task_id] + 1
                            x_train, y_train = train_padded_sentences[task_id][prev_index:index], train_cat_labels[task_id][
                                                                                                  prev_index:index]
                            task_train = train_task_labels[task_id][prev_index:index]
                            s_feature = private_features_train[task_id][prev_index:index]
                    prev_index = index
                    if batch%100==0:
                        print("Batch processing (to train_disc): ", batch)
                print("Epoch completed: ", epoch)
                print("done...")
            return disc_model

        def train_private_features(epochs, batch_size=params.batch_size):
            g_batch_num = [0] * params.N_TASK
            n_batch = int(max_sample_length / batch_size)
            models = {}
            total_losses = {}
            task_losses = {}
            disc_losses = {}
            task_accuracies = {}
            disc_accuracies = {}
            callbacks = {}
            actual_predicted = {}
            best_val_loss = math.inf
            best_f1 = 0.0
            for task_id in range(params.N_TASK):
                models[task_id] = private_model(task_id)
            for epoch in range(epochs):
                print("Epoch started ....", epoch)
                prev_index = 0
                for batch in range(n_batch):
                    index = (batch + 1) * batch_size
                    for task_id in range(params.N_TASK):
                        if index <= train_length[task_id]:
                            g_batch_num[task_id] = g_batch_num[task_id] + 1
                            x_train, y_train = train_padded_sentences[task_id][prev_index:index], train_cat_labels[task_id][
                                                                                                  prev_index:index]
                            task_train = train_task_labels[task_id][prev_index:index]
                            models[task_id].train_on_batch(x_train,
                                                               y_train)

                    prev_index = index
                    if batch%100==0:
                        print("Batch processing: ", batch)

                print("Epoch completed: ", epoch)
                print("done...")
            return models

        def train(epochs, disc_model=None, batch_size=params.batch_size):
            g_batch_num = [0] * params.N_TASK
            n_batch = int(max_sample_length / batch_size)
            models = {}
            total_losses = {}
            task_losses = {}
            disc_losses = {}
            ortho_losses = {}
            task_accuracies = {}
            disc_accuracies = {}
            callbacks = {}
            actual_predicted = {}
            best_val_loss = math.inf
            best_f1 = 0.0
            best_task_f1 ={}
            train_names = ['task_loss', 'task_acc', 'disc_loss',
                           ]
            val_names = ['val_task_loss',
                         'val_task_acc', 'val_disc_loss']

            for task_id in range(params.N_TASK):
                models[task_id] = basic_model(task_id, disc_model)
                best_task_f1[task_id] = [0.0, 0.0, 0.0]
                result_to_store[task_id]  ={}


            for epoch in range(epochs):
                print("Epoch started ....", epoch)
                for task_id in range(params.N_TASK):
                    task_losses[task_id], \
                    disc_losses[task_id], task_accuracies[task_id], ortho_losses[task_id] = [], [], [], []

                prev_index = 0
                total_loss=0.0
                for batch in range(n_batch):
                    index = (batch + 1) * batch_size
                    for task_id in range(params.N_TASK):
                        if index <= train_length[task_id]:
                            g_batch_num[task_id] = g_batch_num[task_id] + 1
                            x_train, y_train = train_padded_sentences[task_id][prev_index:index], train_cat_labels[task_id][
                                                                                                  prev_index:index]
                            task_train = train_task_labels[task_id][prev_index:index]
                            if params.isAdv:
                                local_task_losses, \
                                local_disc_losses, local_task_accuracies = models[
                                    task_id].train_on_batch([x_train, task_train], y_train
                                                            )
                                task_losses[task_id].append(float(local_task_losses))
                                disc_losses[task_id].append(float(local_disc_losses))
                                task_accuracies[task_id].append(float(local_task_accuracies))
                            else:
                                local_task_losses, local_task_accuracies = models[
                                    task_id].train_on_batch([x_train, task_train], y_train
                                                            )
                                task_losses[task_id].append(float(local_task_losses))

                                task_accuracies[task_id].append(float(local_task_accuracies))

                    prev_index = index

                for task_id in range(params.N_TASK):
                    print("Evaluating for task: ", get_task_name(task_id, params.DATASETS))
                    best_res, pre, rec, f1, pred_prob, actual_label = evalaute(task_id, models=models)

                    if f1[0] > best_task_f1[task_id][0]:
                        # print("Best f1 after epoch: ", epoch)
                        result_to_store[task_id]['epoch'] = epoch
                        result_to_store[task_id]['macro'] = best_res[task_id]
                        result_to_store[task_id]['macro-p'] = pre[0]
                        result_to_store[task_id]['macro-r'] = rec[0]
                        result_to_store[task_id]['macro-f'] = f1[0]
                        best_task_f1[task_id][0] = f1[0]

                    if f1[1] > best_task_f1[task_id][1]:
                        # print("Best f1 after epoch: ", epoch)
                        result_to_store[task_id]['epoch'] = epoch
                        result_to_store[task_id]['micro'] = best_res[task_id]
                        result_to_store[task_id]['micro-p'] = pre[1]
                        result_to_store[task_id]['micro-r'] = rec[1]
                        result_to_store[task_id]['micro-f'] = f1[1]
                        best_task_f1[task_id][1] = f1[1]


                    if f1[2] > best_task_f1[task_id][2]:
                        # print("Best f1 after epoch: ", epoch)
                        result_to_store[task_id]['epoch'] = epoch
                        result_to_store[task_id]['weighted'] = best_res[task_id]
                        result_to_store[task_id]['weighted-p'] = pre[2]
                        result_to_store[task_id]['weighted-r'] = rec[2]
                        result_to_store[task_id]['weighted-f'] = f1[2]
                        best_task_f1[task_id][2] = f1[2]



                print("Epoch completed: ", epoch)
                print("done...")

        s_model = shared_model()
        Flip = GradientReversal(1)
        if params.isAdv:
            private_f = train_private_features(10)
            disc_model = train_disc(10, private_f)
            train(params.epoch, disc_model)
        else:
            train(1)
        print('Completed...')
        print('*'*80)
        print("Final result of fold: ", global_fold)
        for task_id, meta_dic in result_to_store.items():
            print(get_task_name(task_id, params.DATASETS))
            for metric, value in meta_dic.items():
                if metric=='micro' or metric =='macro' or metric =='weighted':
                    print('**********'+metric+'*************')
                    for sub_met, sub_value in value.items():

                        print(sub_met)
                        print(sub_value)
                else:
                    print(metric)
                    print(value)
        exp_result[global_fold] = result_to_store

    print("Final result of experiment: ")
    final_results_task = {}
    for task_id in range(params.N_TASK):
        final_results_task[get_task_name(task_id, params.DATASETS)]={}

    for f, best_res in exp_result.items():
        for task_id, meta_dic in best_res.items():
            for metric, value in meta_dic.items():
                if metric in final_results_task[get_task_name(task_id, params.DATASETS)]:
                    final_results_task[get_task_name(task_id, params.DATASETS)][metric].append(value)
                else:
                    final_results_task[get_task_name(task_id, params.DATASETS)][metric]=[value]

    # print(final_results_task)

    for task, fold_res in final_results_task.items():
        print(task)
        print(35*"=")
        avg_prec = 0.0
        avg_rec = 0.0
        for metric, values in fold_res.items():

            if metric=='macro-p' or metric=='macro-r':
                assert len(values)==params.TOTAL_FOLD
                to_be_avg=0.0
                print(metric, end='\t')
                for value in values:
                    to_be_avg += float(value)
                    print(str(value),  end='\t')
                print(float(to_be_avg)/float(params.TOTAL_FOLD))
                if '-p' in metric:
                    avg_prec = float(to_be_avg)/float(params.TOTAL_FOLD)
                if '-r' in metric:
                    avg_rec= float(to_be_avg)/float(params.TOTAL_FOLD)
                print(35*'*')

            # if metric=='micro-p' or metric=='micro-r' or metric=='micro-f':
            #     assert len(values) == params.TOTAL_FOLD
            #     to_be_avg=0.0
            #     print(metric, end='\t')
            #     for value in values:
            #         to_be_avg += float(value)
            #         print(str(value),  end='\t')
            #     print(float(to_be_avg)/float(params.TOTAL_FOLD))
            #     print(35 * '*')
            # if metric=='weighted-p' or metric=='weighted-r' or metric=='weighted-f':
            #     assert len(values) == params.TOTAL_FOLD
            #     to_be_avg=0.0
            #     print(metric, end='\t')
            #     for value in values:
            #         to_be_avg += float(value)
            #         print(str(value),  end='\t')
            #     print(float(to_be_avg)/float(params.TOTAL_FOLD))

            # if metric == 'micro' or metric == 'macro' or metric == 'weighted':
            #     print('**********' + metric + '*************')
            #     for value in values:
            #         for sub_met, sub_value in value.items():
            #             print(sub_met)
            #             print(sub_value)
        macro_f = 2*avg_prec*avg_rec/(avg_prec+avg_rec)
        print("Macro F-Score: ", macro_f)
        print(35*'*')

def get_parser():
    """
        Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Adversarial Training for RE")
    # model parameters
    parser.add_argument("--EMBEDDING_FILE", type=str, default='../data/wikipedia-pubmed-and-PMC-w2v.bin',
                        help="pubmed embedding file path")
    parser.add_argument("--MAX_SEQUENCE_LENGTH", type=int, default=60,
                        help="Maximum length of input sentence")
    parser.add_argument("--MAX_NB_WORDS", type=int, default=10000,
                        help="Maximum number of words in vocabulary")
    parser.add_argument("--EMBEDDING_DIM", type=int, default=200,
                        help="Embedding size")
    parser.add_argument("--d_rate", type=float, default=0.3,
                        help="Dropout")
    parser.add_argument("--HIDDEN_SIZE", type=float, default=64,
                        help="Size of hidden units")
    parser.add_argument("--N_TASK", type=int, default=2,
                        help="No. of task")
    parser.add_argument("--N_GRU", type=int, default=1,
                        help="Number of GRU layers")
    parser.add_argument("--MLP_SIZE", type=int, default=100,
                        help="Hidden size of MLP")
    parser.add_argument("--r_size", type=int, default=5,
                        help="Attention Hyper-parameter")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size to update the weights")
    parser.add_argument("--TOTAL_FOLD", type=int, default=10,
                        help="Number of fold ")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of epochs ")

    parser.add_argument("--isSelfAttention", type=bool, default=True,
                        help="Whether to use attention")
    parser.add_argument("--isAdv", type=bool, default=True,
                        help="Whether to use adversarial training")

    parser.add_argument('--NO_OF_LABELS', action='store',
                        type=int, nargs='*', default=['2', '2', '5', '6'],
                        help="No. of labels in each class")
    parser.add_argument("--training_mode", type=int, default=2,
                        help="training mode")

    return parser


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    if params.training_mode == 1:  ## With aimed and biomed
        params.N_TASK = 2
        params.NO_OF_LABELS = [2, 2]
        params.DATASETS = ['aimed', 'bioinfer']
        params.ENTITY_NAME = [['PROTIEN'], ['PROTIEN']]

    elif params.training_mode == 2:  ## With ddi and i2b2
        params.N_TASK = 2
        params.NO_OF_LABELS = [5, 6]
        params.DATASETS = ['ddi', 'i2b2']
        params.ENTITY_NAME = [['DRUG'], ['BTREATMENT', 'BPROBLEM', 'BTEST']]

    else:   ## with all 4 datasets
        params.N_TASK = 4
        params.NO_OF_LABELS = [2, 2, 5, 6]
        params.DATASETS = ['aimed', 'bioinfer', 'ddi', 'i2b2']
        params.ENTITY_NAME = [['PROTIEN'], ['PROTIEN'], ['DRUG'], ['BTREATMENT', 'BPROBLEM', 'BTEST']]

    main(params)

