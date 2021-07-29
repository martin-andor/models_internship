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
import models

#This is necessary for TF to use the GPU correctly.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#This is to register training in tensorboard.
root_logdir = os.path.join(os.curdir, 'tensorboard_logs')
run_logdir = utils.get_run_logdir(root_logdir)

tensorboard_cb = utils.TBCallback(run_logdir,
                        histogram_freq=1,
                        embeddings_freq=1)

#This is for early stopping with validation set.
earlystopping_cb = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 2,
    verbose = 1,
    mode = 'auto',
    restore_best_weights=True)

#class variables
vocab_size = 20000
embedding_size = 200
BUFFER_SIZE = 10000

#loading and preprocessing the dataset
mpdb_trainval_path = '/home/martin/master_ai/internship/databases/msr/msr_paraphrase_train.txt'
mpdb_test_path = '/home/martin/master_ai/internship/databases/msr/msr_paraphrase_test.txt'

train_val_data, raw_train_ds = utils.load_microsoft_ds(mpdb_trainval_path)
train_val_data = train_val_data.shuffle(BUFFER_SIZE)
val_data = train_val_data.take(int(.15*len(train_val_data)))
train_data = train_val_data.skip(int(.15*len(train_val_data)))
test_data, _ = utils.load_microsoft_ds(mpdb_test_path)

BATCH_SIZE = 64
train_data = train_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#The two first layers of the model are specific to each type of embeddings.
vectorization_layer = layers.experimental.preprocessing.TextVectorization(
                        max_tokens=vocab_size,
                        standardize=utils.lower_keep_punctuation,
                        output_mode='int')

vectorization_layer.adapt(raw_train_ds)

embedding_layer = layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_size,
    mask_zero=True)

first_layers = [vectorization_layer, embedding_layer]

#Loading the model. The options are average, dense, lstm_last, lstm_max_pool, lstm_mean_pool, lstm_attention and cnn.
model_type = 'cnn'
shared_encoder_model = models.load_model(model_type, first_layers, embedding_size)
model = models.complete_model(shared_encoder_model, input='string')

#Compiling the model and fitting it to the data.
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=['accuracy'])

keras.utils.plot_model(model, show_shapes=True,to_file='model1.png')

history=model.fit(train_data,
                    epochs=50,
                    validation_data=val_data,
                    callbacks=[tensorboard_cb, earlystopping_cb])

keras.utils.plot_model(model, show_shapes=True,to_file='model1.png')
model.save('Trained_models/bi-lstm_max_microsoft')

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
