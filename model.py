import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from tensorflow.keras.utils import plot_model


# Define the paths
BASE_DIR = 'D:/Image Caption Generator/Flicker8k_Dataset'
WORKING_DIR = 'D:/Image Caption Generator'

# Check if the image file exists and load it
def load_image(img_path, target_size=(224, 224)):
    try:
        image = load_img(img_path, target_size=target_size)
        image = img_to_array(image)
        return image
    except Exception as e:
        print(f"Error loading image: {img_path}")
        print(e)
        return None

# # Load VGG16 model
# model = VGG16()
# # Restructure the model
# model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# # Extract features from images
# features = {}
# directory = BASE_DIR
# image_files = os.listdir(directory)[:20]  # Select the first 20 images

# for img_name in tqdm(image_files):
#     # Load the image from file
#     img_path = os.path.join(directory, img_name)
#     image = load_image(img_path)
#     if image is not None:
#         # Convert image pixels to numpy array
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         # Preprocess image for VGG
#         image = preprocess_input(image)
#         # Extract features
#         feature = model.predict(image, verbose=0)
#         # Get image ID
#         image_id = img_name.split('.')[0]
#         # Store feature
#         features[image_id] = feature

# # Save the features to a file
# np.save(os.path.join(WORKING_DIR, 'features.npy'), features)

# # Load the features from file
# features = np.load(os.path.join(WORKING_DIR, 'features.npy'), allow_pickle=True).item()

# Open and preprocess the captions file
with open(os.path.join(WORKING_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

# Create mapping of image to captions
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    # Split the line by comma(,)
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]\
    

    # Remove extension from image ID
    image_id = image_id.split('.')[0]
    # Convert caption list to string
    caption = " ".join(caption)
    # Create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # Store the caption
    mapping[image_id].append(caption)

# Preprocess the captions
import re
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # Take one caption at a time
            caption = captions[i]
            # Preprocessing steps
            # Convert to lowercase
            caption = caption.lower()
            # Delete digits, special characters, etc.
            caption = re.sub('[^A-Za-z]', ' ', caption)
            # Delete additional spaces
            caption = re.sub('\s+', ' ', caption)
            # Add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

# Preprocess the captions
clean(mapping)

# Get all captions
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
# Save the tokenizer to a file
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    
# Get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# Create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:

        
        for key in data_keys:
            if key not in features:  # Skip if the key is not in features
                continue
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = [], [], []
                n = 0


# Define the model architecture
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Plot the model
model.summary()

# # Train the model
# epochs = 10
# batch_size = 32
# steps = min(len(train) // batch_size,50)

# # Train the model
# for i in range(epochs):
#     # Create data generator
#     generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
#     # Fit for one epoch
#     model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# # Save the model

# model.save('best_model.h5')
# # Load the pre-trained model
# pretrained_model = load_model('best_model.h5')
