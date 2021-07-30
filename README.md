# models_internship
All the models used for compositionality experiments.

Explanation:

utils.py - contains general helper functions for loading datasets, processing strings etc.

models.py - contains functions for building models. Includes a custom MyAttention(keras.layers.Layer) class and a keras CNN1D model built according to the description in Conneau et al., 2017, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.

training_loaders - Scripts to load trainings on the mpdb dataset. Each type of embeddings (trainable, word2vec, GloVe, Elmo) requires a slightly different approach. The variable model_type determines the kind of model that will be loaded. These scripts are easily modifiable for other datasets.

bert.py - works as a training loader for the bert model, as the loading of the model and the preprocessing of the dataset is very specific.

w2v.py - script to train word2vec embeddings based on a document.

elmoize_ds_to_tfr.py - script to convert a dataset to its elmoized version (with elmo embeddings standing for word tokens) and save it to a TFRecord file. It is needed because otherwise the dataset occupies the whole RAM.
