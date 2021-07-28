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

ELMO_PATH = '/home/martin/master_ai/internship/pretrained_vectors/elmo/nlpl'
embedding_model = ElmoModel()
embedding_model.load(ELMO_PATH, max_batch_size=32)

def elmoize(input, elmo_model):
    output = elmo_model.get_elmo_vectors(input, layers='top')
    return output

def preprocess(sentence):
    exclude = set(string.punctuation)
    sent_proc = ''.join(char for char in sentence if char not in exclude)
    sent_proc = ' '.join(sent_proc.split())
    return sent_proc

def load_elmoized_microsoft_ds(path):
    '''Loading function for the elmoized microsoft paraphrase database.'''
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

    sents1 = elmoize(sents1, embedding_model)
    sents2 = elmoize(sents2, embedding_model)
    print(f'Shape of elmoized sentences 1: {sents1.shape}')
    print(f'Shape of elmoized sentences 2: {sents2.shape}')
    return sents1, sents2, labels

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    '''VALUE MUST BE A LIST!'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint. Value is a list!!!!"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _tensor_to_string(tensor):
     return tf.io.serialize_tensor(tensor)

def serialize_example(sent1_emb: np.ndarray, sent2_emb: np.ndarray, label: int):
    '''Creates a tf.train.SequenceExample message ready to be written to a file.'''
    #dictionary mapping feature name to tf.train.Example compatible data type
    sent1_list = sent1_emb.tolist()
    sent2_list = sent2_emb.tolist()
    sent1_list_of_features = []
    sent2_list_of_features = []
    for embedding in sent1_list: #each embedding is a list of floats.
        embedding_as_feature = _float_feature(embedding)
        sent1_list_of_features.append(embedding_as_feature)
    for embedding in sent2_list: #each embedding is a list of floats.
        embedding_as_feature = _float_feature(embedding)
        sent2_list_of_features.append(embedding_as_feature)

    feature_list = {
        'embeddings1':tf.train.FeatureList(feature=sent1_list_of_features),
        'embeddings2':tf.train.FeatureList(feature=sent2_list_of_features),
        }
    context = tf.train.Features(feature={'label':_int64_feature([label])})
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example_proto = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return example_proto.SerializeToString()

sents1, sents2, labels = load_elmoized_microsoft_ds('/home/martin/master_ai/internship/databases/msr/msr_paraphrase_test.txt')
filename = 'mpdb_test_elmo.tfrecord'

with tf.io.TFRecordWriter(filename) as f:
    for i in range(len(sents1)):
        emb1 = sents1[i]
        emb2 = sents2[i]
        lab = labels[i]
        ser_ex = serialize_example(emb1, emb2, lab)
        f.write(ser_ex)

print(*'\n')
