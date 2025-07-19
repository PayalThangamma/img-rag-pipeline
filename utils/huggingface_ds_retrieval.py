"""
Script to retrieve huggingface dataset

Dataset used: PixelProse
"""
import requests
import pandas as pd

def fetch_x_rows(offset:int, x:int = 100):
    """
    Function to fetch x amount of rows from dataset
    Args:
        x: int - Number of rows where 100 is the maximum
        offset: int - Number of items to skip 

    Return:
        list of dictionaries where each dictionary corresponds to 1 row
    """
    url = f"https://datasets-server.huggingface.co/rows?dataset=tomg-group-umd%2Fpixelprose&config=default&split=train&offset={offset}&length={x}"
    response = requests.get(url)

    if response.status_code == 200:
        # successful response
        return response.json()["rows"]

    else:
        # if response is unsuccessful
        raise Exception(f"Request failed at offset {offset}: {response.status_code} : {response.text}")
    
def save_to_csv(data:list, path: str = "./pixel_prose.csv"):
    """
    Saves data into a retrievable CSV

    Args:
        data: list of rows containing attributes
        path: an output directory specifying where the data should be saved to

    Return: 
        Dataframe before it was saved to CSV
    """

    df = pd.DataFrame(data)

    # saves dataframe to json file
    df.to_csv(path)
    print(f"Successfully written to: {path}")
    return df