import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from tensorflow.python.ops import summary_ops_v2
import pydot
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow_addons as tfa
from official.nlp import optimization
import utils

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

root_logdir = os.path.join(os.curdir, 'tensorboard_logs')
run_logdir = utils.get_run_logdir(root_logdir)

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir,
                                            embeddings_freq=1)

earlystopping_cb = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 2,
    verbose = 1,
    mode = 'auto',
    restore_best_weights=True)

BATCH_SIZE = 64

vocab_size = 20000
embedding_size = 200

BUFFER_SIZE = 50000
train_val_data, raw_train_ds = utils.load_microsoft_ds('/home/martin/master_ai/internship/databases/msr/msr_paraphrase_train.txt')
train_val_data = train_val_data.shuffle(BUFFER_SIZE)
val_data = train_val_data.take(int(.15*len(train_val_data)))
train_data = train_val_data.skip(int(.15*len(train_val_data)))
test_data1, _ = utils.load_microsoft_ds('/home/martin/master_ai/internship/databases/msr/msr_paraphrase_test.txt')
test_data2 = utils.load_paws_ds('/home/martin/master_ai/internship/databases/paws/final/test.tsv', BATCH_SIZE)

train_data = train_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data1 = test_data1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

bert_model_name = 'small_bert/bert_en_uncased_L-2_H-256_A-4'
tfhub_handle_encoder = utils.map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = utils.map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

def make_bert_preprocess_model(sentence_features, seq_length=128):
    '''This model takes the two sentences and combines them into one tensor,
    that clearly distinguishes between their representations. That's why the
    rest of the model is sequential.'''
    input_segments = [
        tf.keras.layers.Input(shape=(),dtype=tf.string, name=ft)
        for ft in sentence_features]

    bert_preprocess = hub.load(tfhub_handle_preprocess)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(segments)
    return tf.keras.Model(input_segments, model_inputs)

bert_preprocess_model = make_bert_preprocess_model(['sentence1','sentence2'])

def build_classifier_model():
    input_segments = [
        tf.keras.layers.Input(shape=(),dtype=tf.string, name=ft)
        for ft in ['sentence1', 'sentence2']]
    inputs = bert_preprocess_model(input_segments)
    encoder = hub.KerasLayer(tfhub_handle_encoder,trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(input_segments, net, name='prediction')

loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.metrics.BinaryAccuracy()

epochs = 50
steps_per_epoch = tf.data.experimental.cardinality(train_data).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

classifier_model = build_classifier_model()
classifier_model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

print(f'Training model with {tfhub_handle_encoder}')

history = classifier_model.fit(
                x=train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=[tensorboard_cb, earlystopping_cb])

loss1, accuracy1 = classifier_model.evaluate(test_data1)

print(f'Loss on microsoft: {loss1}')
print(f'Accuracy on microsoft: {accuracy1}\n')

loss2, accuracy2 = classifier_model.evaluate(test_data2)
print(f'Loss on paws: {loss2}')
print(f'Accuracy on paws: {accuracy2}')

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

saved_model_path = 'trained_models/bert_simple_mpdb'
classifier_model.save(saved_model_path, include_optimizer=False)
bert_preprocess_model.save('trained_models/bert_preprocess')
