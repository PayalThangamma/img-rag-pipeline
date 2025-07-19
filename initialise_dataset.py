from utils.huggingface_ds_retrieval import fetch_x_rows, save_to_csv
from utils.image_retrieval import read_images, clean_csv


import pandas as pd

# we need to request 500 images
TOTAL_NUM_IMAGES = 600
NUM_IMAGES_PER_REQ = 100
DIR_NAME = "./images"
CSV_DIR_NAME="./pixel_prose.csv"
NEW_CSV_DIR_NAME="./image_caption_data.csv"

all_data = []
for offset_val in range(0, TOTAL_NUM_IMAGES, 100):
    
    data = fetch_x_rows(offset_val, NUM_IMAGES_PER_REQ)
    print(f"Retrieved {NUM_IMAGES_PER_REQ} images")
    # add to list 
    all_data += data

all_data_flattened = [data["row"] for data in all_data]
df = save_to_csv(all_data_flattened)


failed_img_extractions = read_images(df, DIR_NAME)
print(f"Number of images which failed to extract: {failed_img_extractions}")

# remove rows for which images could not be extracted
df = clean_csv(failed_img_extractions, df, CSV_DIR_NAME)
print("cleaned dataframe")
# Now, we need to generate augmented captions, save it to the dataframe, and save it to a new directory
df = pd.read_csv("./pixel_prose.csv")