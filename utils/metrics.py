import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
import torch
from transformers import ProcessorMixin, PreTrainedModel
from transformers import BlipProcessor, BlipModel, CLIPModel, AutoProcessor, SiglipModel # Added SiglipModel, AutoProcessor
from tqdm import tqdm # ADDED THIS IMPORT


def iterate_ds(k: int, df: pd.DataFrame, metric: str,
               processor: ProcessorMixin, model: PreTrainedModel,
               caption_column: str, client: QdrantClient, collection_name: str,
               device: str = "cuda",
               use_spectral: bool = False,
               spectral_text_embeddings: np.ndarray = None):

    """
    Iterates through the dataset and computes the metric (MRR or Recall@k)

    Args:
        k (int): Number of top results to consider in the metric
        df (pd.DataFrame): Dataset with image-caption pairs
        metric (str): "mrr" or "recall@k"
        processor (ProcessorMixin): Tokenizer/processor for text
        model (PreTrainedModel): BLIP/CLIP model
        caption_column (str): Column with caption
        client (QdrantClient): Connection to Qdrant
        collection_name (str): Name of Qdrant collection
        device (str): Device to use ("cuda" or "cpu")
        use_spectral (bool): Whether to use spectral embeddings
        spectral_text_embeddings (np.ndarray): Precomputed spectral caption embeddings

    Returns:
        float: Average value of the chosen metric
    """
    total_metric = 0

    model.to(device)
    model.eval()

    for i, row_data in tqdm(df.iterrows(), total=len(df), desc=f"Calculating {metric}"): 
        try:
            img_id = row_data["id"] 

            # Get text embedding
            if use_spectral:
                if spectral_text_embeddings is None:
                    raise ValueError("spectral_text_embeddings must be provided if use_spectral=True")
                caption_embeddings = spectral_text_embeddings[i]
            else:
                caption = row_data[caption_column]
                inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    features = model.get_text_features(**inputs)
                caption_embeddings = (features / features.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().squeeze()

            if metric == "mrr":
                total_metric += mean_reciprocal_rank(k, caption_embeddings, img_id, client, collection_name)
            elif metric == "recall@k":
                total_metric += recall_at_k(k, caption_embeddings, img_id, client, collection_name)

        except Exception as e:
            print(f"Exception occurred at row {i} (ID: {img_id if 'img_id' in locals() else 'N/A'}): {e}")

    return total_metric / len(df)

def mean_reciprocal_rank(k:int, caption_embeddings: np.ndarray, id_to_search: int, client:QdrantClient, collection_name:str):
    """
    This measures the quality of querying by taking the rank of the relevant item queried into the calculation.



    Args:
        k: int - Number of items to query from the vector db (and thus the number of items from which we want to find the rank)
        caption_embeddings: np.ndarray - The embeddings of a caption when passed through a model
        id_to_search: int - An integer identifier used to match an image with a caption when querying
        client: QdrantClient - Reference to the Qdrant Client so that vector database can be accessed
        collection_name: str - Name of the Qdrant collection name so that embeddings can be retrieved.


    """
    search_result = client.query_points(
        collection_name=collection_name,
        query=caption_embeddings,
        with_payload=False,
        limit=k
    ).points

    # obtain rank of item 
    rank_of_searched_item = 0
    for idx, point in enumerate(search_result):
        if point.id == id_to_search:
            rank_of_searched_item = idx+1
            break 

    # ensure division by 0 does not occur
    if rank_of_searched_item == 0:
        return 0
    else:
        return 1/rank_of_searched_item

def recall_at_k(k:int, caption_embeddings: np.ndarray, id_to_search: int, client:QdrantClient, collection_name:str):
    """
    This defines recall at k. This value will either be 0 or 1 as this particular dataset only has 1 relevant item. Therefore, the implementation will be different from the actual recall_at_k to save computation costs

    Args:
        k: int - Number of items to query from the vector db (and thus the number of items from which we want to find the rank)
        caption_embeddings: np.ndarray - The embeddings of a caption when passed through a model
        id_to_search: int - An integer identifier used to match an image with a caption when querying
        client: QdrantClient - Reference to the Qdrant Client so that vector database can be accessed
        collection_name: str - Name of the Qdrant collection name so that embeddings can be retrieved. 

    """

    search_result = client.query_points(
        collection_name=collection_name,
        query=caption_embeddings,
        with_payload=False,
        limit=k
    ).points

    id_list = [item.id for item in search_result]

    if id_to_search in id_list:
        return 1
    else:
        return 0
    
if __name__ == "__main__":

    client = QdrantClient(url="http://localhost:6333")

    df_short = pd.read_csv("../short_generated_captions.csv")
    df_long = pd.read_csv("../long_generated_captions.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        "CLIP": {"id": "openai/clip-vit-base-patch32", "dim": 512},
        "BLIP": {"id": "Salesforce/blip-image-captioning-base", "dim": 768},
        "SIGLIP": {"id": "google/siglip-base-patch16-224", "dim": 768}
    }

    current_model_name = "CLIP"
    model_id = model_configs[current_model_name]["id"]
    model_dim = model_configs[current_model_name]["dim"]

    if current_model_name == "CLIP":
        processor = AutoProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
    elif current_model_name == "BLIP":
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipModel.from_pretrained(model_id).to(device)
    elif current_model_name == "SIGLIP":
        processor = AutoProcessor.from_pretrained(model_id)
        model = SiglipModel.from_pretrained(model_id).to(device)
    else:
        raise ValueError("Selected model not configured.")

    original_collection_name = f"{current_model_name.lower()}_image_db"
    spectral_collection_name = f"{current_model_name.lower()}_spectral_collection"

    k_values = [1, 5, 10]

    datasets = [("Short Captions", df_short), ("Long Captions", df_long)]
    metrics_to_test = ["mrr", "recall@k"]

    for name, df in datasets:
        for metric in metrics_to_test:
            for k_val in k_values:

                avg_score_orig = iterate_ds(
                    k_val, df, metric, processor, model, "paraphrased_caption",
                    client, original_collection_name, device, use_spectral=False
                )
                print(f"Average {metric.upper()}@{k_val} for ORIGINAL embeddings: {avg_score_orig:.4f}")
