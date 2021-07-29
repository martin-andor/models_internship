import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from tensorflow.python.ops import summary_ops_v2
import pydot
import utils

class MyAttention(keras.layers.Layer):
    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        self.W_keys = self.add_weight(
            name='convert_to_keys',
            shape=[batch_input_shape[-1], batch_input_shape[-1]],
            initializer='glorot_normal')
        self.W_keys = tf.broadcast_to(self.W_keys, [batch_input_shape[1],
                                                batch_input_shape[-1],
                                                batch_input_shape[-1]])
        self.v1 = self.add_weight(
            name='perspective_v1',
            shape=[batch_input_shape[-1],1],
            initializer='glorot_normal')
        self.v2 = self.add_weight(
            name='perspective_v2',
            shape=[batch_input_shape[-1],1],
            initializer='glorot_normal')
        self.v3 = self.add_weight(
            name='perspective_v3',
            shape=[batch_input_shape[-1],1],
            initializer='glorot_normal')
        self.v4 = self.add_weight(
            name='perspective_v4',
            shape=[batch_input_shape[-1],1],
            initializer='glorot_normal')
        super().build(batch_input_shape)

    def call(self, X):
        #First, we need to obtain the keys by multiplying the timesteps by W_keys
        #This should be equivalent to processing it by a time-distributed dense layer with dim neurons
        #self.W_keys = tf.broadcast_to(self.W_keys, [X.shape.as_list()[1],
                                        #        X.shape.as_list()[-1],
                                        #        X.shape.as_list()[-1]])
        self.keys = tf.matmul(self.W_keys,tf.expand_dims(X, -1))
        self.keys = tf.squeeze(self.keys,axis=-1)

        self.scores1 = tf.matmul(self.keys, self.v1)
        self.weights1 = tf.nn.softmax(self.scores1, axis=1)
        self.sent_embedding1 = tf.reduce_sum(tf.math.multiply(self.weights1, X), axis=1)

        self.scores2 = tf.matmul(self.keys, self.v2)
        self.weights2 = tf.nn.softmax(self.scores2, axis=1)
        self.sent_embedding2 = tf.reduce_sum(tf.math.multiply(self.weights2, X), axis=1)

        self.scores3 = tf.matmul(self.keys, self.v3)
        self.weights3 = tf.nn.softmax(self.scores3, axis=1)
        self.sent_embedding3 = tf.reduce_sum(tf.math.multiply(self.weights3, X), axis=1)

        self.scores4 = tf.matmul(self.keys, self.v4)
        self.weights4 = tf.nn.softmax(self.scores4, axis=1)
        self.sent_embedding4 = tf.reduce_sum(tf.math.multiply(self.weights4, X), axis=1)

        self.sent_embedding = tf.concat([self.sent_embedding1,
                                            self.sent_embedding2,
                                            self.sent_embedding3,
                                            self.sent_embedding4], axis=1)
        return self.sent_embedding

    def compute_output_shape(self,batch_input_shape):
        return tf.TensorShape([batch_input_shape.as_list()[0], batch_input_shape.as_list()[-1]*4])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'activation': keras.activations.serialize(self.activation)}

def build_cnn_model(first_layers, embedding_size):
    if len(first_layers) == 1:
        #This happens when you have elmo embeddings and only masking layer is applied.
        input = keras.Input(shape=(30,1024), dtype=tf.float32, name='sentence')
        embedding_layer = first_layers[0](input)
    else:
        input = keras.Input(shape=(), dtype=tf.string, name='sentence')
        preprocessed_input = first_layers[0](input)
        embedding_layer = first_layers[1](preprocessed_input)
    #1st CNN layer
    conv_layer_1 = layers.Conv1D(embedding_size, 4, activation='relu', padding='same')(embedding_layer)
    max_pool_layer_1 = layers.GlobalMaxPool1D()(conv_layer_1)
    #2nd CNN layer
    conv_layer_2 = layers.Conv1D(embedding_size, 4, activation='relu', padding='same')(conv_layer_1)
    max_pool_layer_2 = layers.GlobalMaxPool1D()(conv_layer_2)
    #3rd CNN layer
    conv_layer_3 = layers.Conv1D(embedding_size, 4, activation='relu', padding='same')(conv_layer_2)
    max_pool_layer_3 = layers.GlobalMaxPool1D()(conv_layer_3)
    #4rd CNN layer
    conv_layer_4 = layers.Conv1D(embedding_size, 4, activation='relu', padding='same')(conv_layer_3)
    max_pool_layer_4 = layers.GlobalMaxPool1D()(conv_layer_4)
    #Final layer
    final_shared_layer = layers.Concatenate()([max_pool_layer_1,
                                                max_pool_layer_2,
                                                max_pool_layer_3,
                                                max_pool_layer_4])

    return keras.Model(inputs=input, outputs=final_shared_layer)

def load_model(model_type, first_layers, embedding_size):
    lstm_settings = {'dropout':0.2, 'recurrent_dropout':0.2}
    if model_type == 'cnn':
        return build_cnn_model(first_layers, embedding_size)

    final_layer_dic = {
        'average': [
            layers.GlobalAveragePooling1D()],
        'dense': [
            layers.Flatten(),
            layers.Dense(
                embedding_size,
                activation='relu')],
        'lstm_last': [
            layers.Bidirectional(
                layers.LSTM(
                    embedding_size,
                    **lstm_settings))],
        'lstm_max_pool': [
            layers.Bidirectional(
                layers.LSTM(
                    embedding_size,
                    return_sequences=True,
                    **lstm_settings)),
            layers.GlobalMaxPool1D()],
        'lstm_mean_pool': [
            layers.Bidirectional(
                layers.LSTM(
                    embedding_size,
                    return_sequences=True,
                    **lstm_settings)),
            layers.GlobalAveragePooling1D()],
        'lstm_attention': [
            layers.Bidirectional(
                layers.LSTM(
                    embedding_size,
                    return_sequences=True,
                    **lstm_settings)),
            MyAttention()],
        }
    final_layers = final_layer_dic[model_type]
    model_layers = first_layers + final_layers
    model = keras.Sequential(model_layers)
    return model

def complete_model(shared_model, input):
    if input == 'string':
        input_1 = keras.Input(shape=(),dtype=tf.string,name='sentence1')
        input_2 = keras.Input(shape=(),dtype=tf.string,name='sentence2')
    else:
        input_1 = keras.Input(shape=(30,1024), dtype=tf.float32, name='sentence1')
        input_2 = keras.Input(shape=(30,1024), dtype=tf.float32, name='sentence2')

    encoder1 = shared_model(input_1)
    encoder2 = shared_model(input_2)

    x = layers.Multiply()([encoder1, encoder2])
    y = layers.Subtract()([encoder1, encoder2])
    z = layers.Concatenate()([encoder1, encoder2, x, y])
    final_layer = layers.Dense(1)(z)

    model = keras.Model(inputs=[input_1,input_2], outputs=final_layer)
    return model
