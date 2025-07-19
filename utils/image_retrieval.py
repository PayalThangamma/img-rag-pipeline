"""
Retrieve images through CSV and save them into the images folder
"""
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import os

def read_images(df: pd.DataFrame, output_dir: str):
    """
    Reads all image URLs from df and saves them into output_dir 

    Args:
        df: pd.DataFrame - A pandas Dataframe containing the "url" key to extract image
        output_dir: str - directory to save images to

    Returns:
        list of df indexes for which image could not be retrieved
    """
    # if directory does not exist create one
    os.makedirs(output_dir, exist_ok=True)

    # extract image URLs and process
    all_img_urls = df["url"]
    failed_extractions = []
    for i in range(len(all_img_urls)):
        try:
            response = requests.get(all_img_urls[i])
            # open image and save
            # convert to RGB 
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # save images
            img.save(os.path.join(output_dir, f"img_{i}.jpg"))
        except Exception as e:
            print(f"Image {i} could not be downloaded: {e}")
            failed_extractions.append(i)

    print(f"Images successfully saved")
    return failed_extractions

def clean_csv(failed_extractions: list, df: pd.DataFrame, path: str = "./pixel_prose.csv"):
    """
    Takes in a list of indexes for which an image cannot be saved, and removes those rows from the pandas dataframe df

    Args:
        failed_extractions: list containing indexes of rows for which image could not be extracted
        df: Pandas Dataframe of all rows
        path: an output directory specifying where the data should be saved to

    Return:
        df: Cleaned dataframe which has rows for which images are already extracted
    """

    df = df.drop(failed_extractions)
    df.to_csv(path)

    return df