import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
from tensorflow.python.ops import summary_ops_v2
import pydot
import utils
import string
import models
import gc

#This is necessary for TF to use the GPU correctly.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#This is to register the training in TensorBoard
root_logdir = os.path.join(os.curdir, 'tensorboard_logs')
run_logdir = utils.get_run_logdir(root_logdir)

tensorboard_cb = utils.TBCallback(run_logdir,
                        histogram_freq=1,
                        embeddings_freq=1)

#This is for early stopping with a validation set
earlystopping_cb = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 2,
    verbose = 1,
    mode = 'auto',
    restore_best_weights=True)

#Class variables
BATCH_SIZE = 1
embedding_size = 200
vocab_size = 20000
BUFFER_SIZE = 5000

#Loading the dataset which has previously been elmoized, that is, the words have changed to elmo embeddings.
mpdb_elmoized_trainval_path = 'mpdb_trainval_elmo.tfrecord'
mpdb_elmoized_test_path = 'mpdb_test_elmo.tfrecord'
train_val_data = utils.load_elmoized_dataset(mpdb_elmoized_trainval_path)
test_data = utils.load_elmoized_dataset(mpdb_elmoized_test_path)

print(f'Train_val_dataset: {train_val_data}')
train_val_data = train_val_data.shuffle(BUFFER_SIZE)

n_examples = 4076
length_val_data = int(.15*n_examples)
length_train_data = n_examples - length_val_data

val_data = train_val_data.take(length_val_data).repeat()
train_data = train_val_data.skip(length_val_data)

train_data = train_data.shuffle(BUFFER_SIZE).repeat()
test_data = test_data.prefetch(tf.data.AUTOTUNE)

print(f'Train dataset after batching: {train_data}')
train_data = train_data.prefetch(tf.data.AUTOTUNE)

#Loading the model. The options are average, dense, lstm_last, lstm_max_pool, lstm_mean_pool, lstm_attention and cnn.
model_type = 'cnn'
first_layers = [layers.Masking()]
shared_encoder_model = models.load_model(model_type, first_layers, embedding_size)
model = models.complete_model(shared_encoder_model, input='elmo')

gc.collect()

#Compiling and fitting the model to the data.
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=['accuracy'])

keras.utils.plot_model(model, show_shapes=True,to_file='model_elmo_cnn.png')

history=model.fit(train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=[earlystopping_cb],
            steps_per_epoch=length_train_data,
            validation_steps=length_val_data)

#model.save('Trained_models/elmo') - for some reason, elmo models do not save.

test_loss, test_acc = model.evaluate(test_data)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

#Plotting the training 
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
utils.plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
utils.plot_graphs(history, 'loss')
plt.ylim(0,None)
plt.show()
