from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras import backend as K
import numpy as np
import h5py
from FeatureFile import features
loaded_features=features()
file_name= './captions.txt'

# Load the captions from the file
with open(file_name, 'r') as file:
    captions_doc = file.read()
    print(captions_doc)

# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # print(caption)
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # # convert caption list to string
    caption = " ".join(caption)
    # print(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # # store the caption
    mapping[image_id].append(caption)
print(mapping)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
image_ids = list(mapping.keys())
image_ids=image_ids[1:]
#
split = int(len(image_ids) * 0.90)
print(split)
train = image_ids[:split]
test = image_ids[split:]
#
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                # print(seq)
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]  # Right padding
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X1.append(features[key].flatten())
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0

# # If the model already exists in memory, delete it
# if 'model' in locals():
#     del model  # Remove reference to the previous model
#     K.clear_session()
# encoder model
# # image feature layers
# inputs1 = Input(shape=(4096,), name="image")
# fe1 = Dropout(0.4)(inputs1)
# fe2 = Dense(256, activation='relu')(fe1)
# # sequence feature layers
# inputs2 = Input(shape=(max_length,), name="text")
# se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
# se2 = Dropout(0.4)(se1)
# se3 = LSTM(256, use_bias=True, recurrent_activation='sigmoid')(se2)
#
# # decoder model
# decoder1 = add([fe2, se3])
# decoder2 = Dense(256, activation='relu')(decoder1)
# outputs = Dense(vocab_size, activation='softmax')(decoder2)
#
# model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# # train the model
# epochs = 5
# batch_size = 32
# steps = len(train) // batch_size
# #
# for i in range(epochs):
#     # create data generator
#     generator = data_generator(train, mapping, loaded_features, tokenizer, max_length, vocab_size, batch_size)
#     # fit for one epoch
#     model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#
# model.save('my_model.keras')
# print("hi")
import keras
model = keras.models.load_model('my_model.keras')

# Load the saved model file
# model_file = 'my_model.h5'
#
# my_model = load_model('my_model.h5')
# print("hi")
# print(my_model)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # Ensure the image has the right shape
    image = np.expand_dims(image, axis=0)  # Add batch dimension
#
    # Initialize the sequence with the start token
    in_text = 'startseq'

    for i in range(max_length):
        # Encode and pad the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
#
        # Predict the next word
        # image = image.flatten()
        yhat = model.predict([image, sequence], verbose=0)

        # Get the predicted word index
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        # Stop if the word is None or the end token is predicted
        if word is None or word == 'endseq':
            break

        in_text += " " + word

    return in_text

from nltk.translate.bleu_score import corpus_bleu
#  validate with test data
actual, predicted = list(), list()
#
for key in tqdm(test):
    # Check if the image ID exists in features
    if key in loaded_features:
        # get actual caption
        captions = mapping[key]
        # predict the caption for image
        y_pred = predict_caption(model, loaded_features[key].flatten(), tokenizer, max_length)
        # split into words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        # append to the list
        actual.append(actual_captions)
        predicted.append(y_pred)
#0-=
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))