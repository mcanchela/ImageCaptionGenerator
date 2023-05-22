import os
import pandas as pd

# The location of the Flickr8K images
dir_Flickr_jpg = "D:/Image Caption Generator/Flicker8k_Dataset"
# The location of the caption file
dir_Flickr_text = "D:/Image Caption Generator/captions.txt"

jpgs = os.listdir(dir_Flickr_jpg)
print("The number of jpg files in Flicker8k: {}".format(len(jpgs)))

# Load captions from the file
df_txt = pd.read_csv(dir_Flickr_text)

# Rename the column to "filename"
df_txt = df_txt.rename(columns={"image": "filename"})

# Add start and end tokens to captions
df_txt['caption'] = df_txt['caption'].apply(lambda x: '<start> ' + x + ' <end>')

uni_filenames = df_txt.filename.unique()
print("The number of unique file names: {}".format(len(uni_filenames)))
print("The distribution of the number of captions for each image:")
print(df_txt.groupby('filename').count()['caption'].value_counts())
print(df_txt.head(50))
