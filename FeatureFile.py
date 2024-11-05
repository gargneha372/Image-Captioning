import os
import pickle
import numpy
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# model=VGG16()
# model=Model(inputs=model.inputs,outputs=model.layers[-2].output)
# print(model.summary())
#
# # Dictionary to store extracted features
# features = {}
# directory = './Images/'
# cnt =0
# # Loop through each item in the directory
# for img_name in tqdm(os.listdir(directory)):
#     img_path = os.path.join(directory, img_name)
#     # Check if the item is a file (not a directory)
#     if os.path.isfile(img_path):
#     #     # Load the image from the file
#         image = load_img(img_path, target_size=(224, 224))
#
#     #     # Convert image pixels to a numpy array
#         image = img_to_array(image)
#         # print(image.shape)
#
#     # #     # Reshape data for the model
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         # print(image2.shape)
#     # #     # Preprocess the image for the model
#         image = preprocess_input(image)
# #         # print(image2.shape)
# #
# #     # #     # Extract features using the model
#         feature = model.predict(image, verbose=0)
#         # print(feature.type())
#
#     # #     # Get image ID (name without extension)
#         image_id = img_name.split('.')[0]
#
#     #     # Store the extracted feature in the dictionary
#         features[image_id] = feature
#         print(features)
#     # cnt = cnt + 1
#     # if cnt == 5:
#     # break
#
# with open('./features.pkl', 'wb') as file:
#     pickle.dump(features, file)
def features():
    with open('./features.pkl', 'rb') as file:
        loaded_features = pickle.load(file)
    return(loaded_features)
