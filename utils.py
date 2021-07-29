import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.python.ops import summary_ops_v2
import matplotlib.pyplot as plt
import string
from simple_elmo import ElmoModel

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def load_elmoized_dataset(filename):
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print(f'Raw loaded dataset: {raw_dataset}')
    dataset = raw_dataset.map(prepare_dataset_for_training)
    return dataset

def prepare_dataset_for_training(example):
    context_features = {
        'label': tf.io.FixedLenFeature([],tf.int64)}
    sequence_features = {
        'embeddings1': tf.io.VarLenFeature(tf.float32),
        'embeddings2': tf.io.VarLenFeature(tf.float32)}
    parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
        example,
        context_features=context_features,
        sequence_features=sequence_features)
    emb1 = tf.RaggedTensor.from_sparse(parsed_feature_lists['embeddings1'])
    emb1 = tf.reshape(emb1.to_tensor(), shape=(1,30,1024))
    emb2 = tf.RaggedTensor.from_sparse(parsed_feature_lists['embeddings2'])
    emb2 = tf.reshape(emb2.to_tensor(), shape=(1,30,1024))
    label = tf.expand_dims(parsed_context['label'], axis=0)
    return ({'sentence1': emb1, 'sentence2': emb2}, label)

def preprocess(sentence):
    exclude = set(string.punctuation)
    sent_proc = ''.join(char for char in sentence if char not in exclude)
    sent_proc = ' '.join(sent_proc.split())
    return sent_proc

def lower_keep_punctuation(input_data):
    '''For TextVectorization: I want lowercase,
    but do not want to strip punctuation, which may be relevant
    for long-distance dependencies.'''
    return tf.strings.lower(input_data)

def load_microsoft_ds(path):
    '''Loading function for the microsoft paraphrase database. ELMO VERSION.
    Also returns a raw dataset with both sentences for textvectorization purposes.'''
    print(f'Processing {path}...')
    with open(path) as f:
        lines = f.readlines()
    c = 0
    pos = 0
    labels = []
    sents1 = []
    sents2 = []
    sents = []
    max_length = 0
    for line in lines:
        if c > 0:
            entries = line.split('\t')
            label = int(entries[0])
            if label == 1:
                pos += 1
            labels.append(label)
            sent1 = preprocess(entries[3])
            sent2 = preprocess(entries[4])
            sent1_elmo = sent1.split()
            sent2_elmo = sent2.split()
            max_length = max(len(sent1.split()), len(sent2.split()), max_length)
            sents1.append(sent1_elmo)
            sents2.append(sent2_elmo)
            sents.append(sent1 + ' ' + sent2)
        c+=1
    print(f'Total lines processed: {c}.')
    print(f'Total positive labels: {pos}. Ratio = {pos/c}')
    print(f'Length sents1 : {len(sents1)}')
    print(f'Length sents2 : {len(sents2)}')
    print(f'Length labels : {len(labels)}')
    print(f'Max length of a sentence : {max_length}')

    sents1 = tf.expand_dims(tf.constant(sents1, dtype=tf.string),axis=-1)
    sents2 = tf.expand_dims(tf.constant(sents2, dtype=tf.string),axis=-1)
    print(f'Shape of sents1 as tensor: {sents1.shape}')
    print(f'Shape of sents2 as tensor: {sents2.shape}')
    ds = tf.data.Dataset.from_tensor_slices(
        ({'sentence1': sents1,
        'sentence2': sents2},
        tf.expand_dims(labels,-1)))
    ds_raw = tf.data.Dataset.from_tensor_slices(
        (tf.expand_dims(sents,-1)))
    return ds, ds_raw

def load_paws_ds(path, batch_size, shuffle_buffer_size=50000):
    '''Loading function for the PAWS dataset.'''
    df = pd.read_csv(path, sep='\t')
    labels = df.pop('label')
    df.drop('id', inplace=True, axis=1)
    sents1 = tf.expand_dims(df.values[:,0],-1)
    sents2 = tf.expand_dims(df.values[:,1],-1)
    tar = tf.expand_dims(labels.values,-1)
    ds = tf.data.Dataset.from_tensor_slices((
        {'sentence1': sents1,
        'sentence2': sents2},
        tar))
    ds = ds.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def plot_graphs(history, metric):
  '''Plots graphs of training NNs'''
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

class TBCallback(tf.keras.callbacks.TensorBoard):
    '''This is necessary for the Tensorboard callback to work with the
    experimental preprocessing layer.'''
    def _log_weights(self, epoch):
        with self._train_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                for layer in self.model.layers:
                    for weight in layer.weights:
                        if hasattr(weight, "name"):
                            weight_name = weight.name.replace(':', '_')
                            summary_ops_v2.histogram(weight_name, weight, step=epoch)
                            if self.write_images:
                                self._log_weight_as_image(weight, weight_name, epoch)
                self._train_writer.flush()

def get_run_logdir(root_logdir):
    '''This is to create the name for the tensorboard log.'''
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)

def load_w2v_embeddings(path_words, path_embeddings):
    dfw = pd.read_csv(path_words,sep='\t', header=None)
    dfe = pd.read_csv(path_embeddings,sep='\t', header=None)
    emb_dic = {}
    emb_size = 0
    for i in range(len(dfw)):
        word = dfw.iat[i,0]
        emb = np.asarray(dfe.iloc[i], dtype='float32')
        new_emb_size = len(emb)
        if new_emb_size != emb_size:
            emb_size = new_emb_size
        emb_dic[word] = emb
    return emb_dic, emb_size

def get_embedding_matrix(embedding_size, vocab_size, word_index, embedding_dict):
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_word_index(text_vec_layer):
    vocabulary = text_vec_layer.get_vocabulary()
    index = dict(zip(vocabulary, range(len(vocabulary))))
    return index

def load_glove_embeddings(path):
    '''Loads the embeddings.
    This functions changes based on the format of the data.'''
    emb_index = {}
    with open(path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            emb_index[word] = coefs
    return emb_index

def load_merged_ds(path_micr, path_paws_train, path_paws_val, batch_size, buffer_size=50000):
    print(f'Processing {path_micr}...')
    with open(path_micr) as f:
        lines = f.readlines()
    c = 0
    labels_ms = []
    sents1_ms = []
    sents2_ms = []
    for line in lines:
        if c > 0:
            entries = line.split('\t')
            labels_ms.append(int(entries[0]))
            sents1_ms.append(entries[3])
            sents2_ms.append(entries[4])
        c+=1
    print(f'Total lines processed: {c}.')
    print(f'Length sents1 : {len(sents1_ms)}')
    print(f'Length sents2 : {len(sents2_ms)}')
    print(f'Length labels : {len(labels_ms)}')

    print(f'Processing {path_paws_train}...')
    with open(path_paws_train) as f:
        lines = f.readlines()
    c = 0
    labels_pt = []
    sents1_pt = []
    sents2_pt = []
    for line in lines:
        if c > 0:
            entries = line.split('\t')
            labels_pt.append(int(entries[3]))
            sents1_pt.append(entries[1])
            sents2_pt.append(entries[2])
        c+=1
    print(f'Total lines processed: {c}.')
    print(f'Length sents1 : {len(sents1_pt)}')
    print(f'Length sents2 : {len(sents2_pt)}')
    print(f'Length labels : {len(labels_pt)}')

    print(f'Processing {path_paws_val}...')
    with open(path_paws_val) as f:
        lines = f.readlines()
    c = 0
    labels_pv = []
    sents1_pv = []
    sents2_pv = []
    for line in lines:
        if c > 0:
            entries = line.split('\t')
            labels_pv.append(int(entries[3]))
            sents1_pv.append(entries[1])
            sents2_pv.append(entries[2])
        c+=1
    print(f'Total lines processed: {c}.')
    print(f'Length sents1 : {len(sents1_pv)}')
    print(f'Length sents2 : {len(sents2_pv)}')
    print(f'Length labels : {len(labels_pv)}')

    print('Merging datasets...')
    labels = np.array(labels_ms + labels_pt + labels_pv)
    labels = tf.expand_dims(labels,-1)
    print(f'Shape of labels: {labels.shape}')

    sents1 = np.array(sents1_ms + sents1_pt + sents1_pv)
    sents1 = tf.expand_dims(sents1, -1)
    print(f'Shape of sents1: {sents1.shape}')

    sents2 = np.array(sents2_ms + sents2_pt + sents2_pv)
    sents2 = tf.expand_dims(sents2,-1)
    print(f'Shape of sents2: {sents1.shape}')

    ds = tf.data.Dataset.from_tensor_slices(
        ({'sentence1': sents1,
        'sentence2': sents2},
        labels))
    ds = ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}
