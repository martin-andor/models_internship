import io
import re
import string
import tensorflow as tf
import tqdm
import string
from os import system
import pandas as pd

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def get_vocabulary_and_longest_sentence(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    d, vocab, index = {},{},1
    vocab['<pad>'] = 0
    length_longest = 0
    c = 0
    for line in lines:
        if c > 0:
            splitline = line.split('\t')
            tokens1 = remove_punctuation_and_split(splitline[3])
            tokens2 = remove_punctuation_and_split(splitline[4])
            length_longest = max(length_longest, len(tokens1), len(tokens2))
            tokens = tokens1 + tokens2
            for token in tokens:
                d[token.strip()] = d.get(token.strip(), 0) + 1
        c += 1
    l = sorted(list(d.items()), reverse=True, key=lambda x:x[1])
    for el in l:
        vocab[el[0]] = index
        index += 1
    print(f'First 5 entries from the dictionary: {list(vocab.items())[:5]}')
    print('Length longest sentence :', length_longest)
    return vocab, length_longest

def remove_punctuation_and_split(sentence: str)->list:
    sentence_split = sentence.lower().split()
    for i in range(len(sentence_split)):
        try:
            if sentence_split[i].strip() in string.punctuation:
                del sentence_split[i]
            else:
                sentence_split[i] = sentence_split[i].strip()
        except: break
    return sentence_split

def pad_to_length(sentence:list, length: int)->list:
    while len(sentence) < length:
        sentence.append(0)
    return sentence

def vectorize_sentences(filepath, vocab, length):
    all_sentences_vectorized = []
    with open(filepath) as f:
        lines = f.readlines()
    c = 0
    for line in lines:
        if c > 0:
            splitline = line.split('\t')
            sentence1 = remove_punctuation_and_split(splitline[3])
            sentence2 = remove_punctuation_and_split(splitline[4])
            sentence1_vectorized = pad_to_length([vocab[word] for word in sentence1],
                                    length)
            sentence2_vectorized = pad_to_length([vocab[word] for word in sentence2],
                                    length)
            all_sentences_vectorized.extend([sentence1_vectorized, sentence2_vectorized])
        c += 1
    print('Number of sentences: ', len(all_sentences_vectorized))
    print('First 10 vectorized sentences: ', all_sentences_vectorized[:10])
    return all_sentences_vectorized

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype='int64'),1)
            negative_sampling_candidates, _,_ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=SEED,
                name='negative_sampling')
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates],0)
            label = tf.constant([1] + [0]*num_ns, dtype='int64')

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    return targets, contexts, labels

class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(vocab_size, embedding_dim, input_length=1,name='w2v_embedding')
        self.context_embedding = Embedding(vocab_size, embedding_dim, input_length=num_ns+1)
        self.dots = Dot(axes=(3,2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

path = '/home/martin/master_ai/internship/databases/msr/msr_paraphrase_train.txt'
#df = pd.read_csv(path, sep='\t')


vocab, length_longest = get_vocabulary_and_longest_sentence(path)
vocab_size = len(vocab)
inverse_vocab = {index: word for word, index in vocab.items()}
all_sentences_vectorized = vectorize_sentences(path, vocab, length_longest)

targets, contexts, labels = generate_training_data(
    sequences=all_sentences_vectorized,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

print('Lengths of data: ', len(targets), len(contexts), len(labels))

BATCH_SIZE = 1024
BUFFER_SIZE = 10000

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

print('Dataset: ', dataset)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 200
word2vec = Word2Vec(vocab_size, embedding_dim, num_ns=4)
word2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
system('tensorboard --logdir logs')
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

out_v = io.open('vectors.tsv', 'w', encoding = 'utf-8')
out_m = io.open('metadata.tsv', 'w', encoding = 'utf-8')

for word, index in vocab.items():
    if index == 0:
        continue
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')
out_v.close()
out_m.close()

try:
    from google.colab import files
    files.download('vectors.tsv')
    files.download('metadata.tsv')
except Exception:
    pass
