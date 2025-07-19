import math
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http import models as rest_models

def create_collections(dimension):
        client = QdrantClient(url="http://localhost:6333")
        client.delete_collection(collection_name="image_db")

        # initialise collection
        client.create_collection(
            collection_name="image_db",
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

        client.delete_collection(collection_name="spectral_collection")

        # initialise collection
        client.create_collection(
            collection_name="spectral_collection",
            vectors_config=VectorParams(size=2 ** (math.floor(math.log2(dimension)) - 2), distance=Distance.COSINE),
        )