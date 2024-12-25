from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer 
from transformers import AutoModel, AutoImageProcessor
import torch
import os
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import uuid
from PIL import Image
from fastembed import SparseTextEmbedding
import cohere

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer("sentence-transformers/LaBSE").to(device)
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
image_encoder = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
qdrant_client = QdrantClient("http://localhost:6333")
sparse_encoder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
co = cohere.ClientV2(os.getenv("cohere_api_key"))

dataset = load_dataset("Karbo31881/Pokemon_images")
ds = dataset["train"]
labels = ds["text"]

def get_sparse_embedding(text: str, model: SparseTextEmbedding):
    embeddings = list(model.embed(text))
    vector = {f"sparse-text": models.SparseVector(indices=embeddings[0].indices, values=embeddings[0].values)}
    return vector

def get_query_sparse_embedding(text: str, model: SparseTextEmbedding):
    embeddings = list(model.embed(text))
    query_vector = models.NamedSparseVector(
        name="sparse-text",
        vector=models.SparseVector(
            indices=embeddings[0].indices,
            values=embeddings[0].values,
        ),
    )
    return query_vector

def upload_text_to_qdrant(client: QdrantClient, collection_name: str, encoder: SentenceTransformer, text: str, point_id_dense: int, point_id_sparse: int):
    try:
        docs = {"text": text}
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id_dense,
                    vector={f"dense-text": encoder.encode(docs["text"]).tolist()},
                    payload=docs,
                )
            ],
        )
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id_sparse,
                    vector=get_sparse_embedding(docs["text"], sparse_encoder),
                    payload=docs,
                )
            ],
        )
        return True
    except Exception as e:
        return False
    
def upload_images_to_qdrant(client: QdrantClient, collection_name: str, vectorsfile: str, labelslist: list):
    try:
        vectors = np.load(vectorsfile)
        docs = []
        for label in labelslist:
            docs.append({"label": label})
        client.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=vectors[idx].tolist(),
                    payload=doc,
                )
                for idx, doc in enumerate(docs)
            ],
        )
        return True
    except Exception as e:
        return False

class SemanticCache:
    def __init__(self, client: QdrantClient, text_encoder: SentenceTransformer, collection_name: str, threshold: float = 0.75):
        self.client = client
        self.text_encoder = text_encoder
        self.collection_name = collection_name
        self.threshold = threshold
    def upload_to_cache(self, question: str, answer: str):
        docs = {"question": question, "answer": answer}
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=self.text_encoder.encode(docs["question"]).tolist(),
                    payload=docs,
                )
            ],
        )
    def search_cache(self, question: str, limit: int = 5):
        vector = self.text_encoder.encode(question).tolist()
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=limit,
        )
        payloads = [hit.payload["answer"] for hit in search_result if hit.score > self.threshold]
        if len(payloads) > 0:
            return payloads[0]
        else:
            return ""


class NeuralSearcher:
    def __init__(self, text_collection_name: str, image_collection_name: str, client: QdrantClient, text_encoder: SentenceTransformer , image_encoder: AutoModel, image_processor: AutoImageProcessor, sparse_encoder: SparseTextEmbedding):
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.qdrant_client = client
        self.sparse_encoder = sparse_encoder

    def search_text(self, text: str, limit: int = 5):
        vector = self.text_encoder.encode(text).tolist()

        search_result_dense = self.qdrant_client.search(
            collection_name=self.text_collection_name,
            query_vector=models.NamedVector(name="dense-text", vector=vector),
            query_filter=None,
            limit=limit,
        )

        search_result_sparse = self.qdrant_client.search(
            collection_name=self.text_collection_name,
            query_vector=get_query_sparse_embedding(text, self.sparse_encoder),
            query_filter=None,
            limit=limit,
        )
        payloads = [hit.payload["text"] for hit in search_result_dense]
        payloads += [hit.payload["text"] for hit in search_result_sparse]
        return payloads
    def reranking(self, text: str, search_result: list):
        results = co.rerank(model="rerank-v3.5", query=text, documents=search_result, top_n = 3)
        ranked_results = [search_result[results.results[i].index] for i in range(3)]
        return ranked_results
    def search_image(self, image: str, limit: int = 5):
        img = Image.open(image)
        inputs = self.image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.image_encoder(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        search_result = self.qdrant_client.search(
            collection_name=self.image_collection_name,
            query_vector=outputs[0].tolist(),
            query_filter=None,
            limit=limit,
        )
        payloads = [f"- {hit.payload['label']} with score {hit.score}" for hit in search_result]
        return payloads


qdrant_client.recreate_collection(
    collection_name="pokemon_texts",
    vectors_config={"dense-text": models.VectorParams(
        size=768,  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    )},
    sparse_vectors_config={"sparse-text": models.SparseVectorParams(
        index=models.SparseIndexParams(
            on_disk=False
        )
    )}
)
textdata = load_dataset("wanghaofan/pokemon-wiki-captions")
names = textdata["train"]["name_en"]
texts = textdata["train"]["text_en"]

c = 0

for j in range(len(texts)):
    txt = names[j].upper() + "\n\n" + texts[j]
    l = c+1
    upload_text_to_qdrant(qdrant_client, "pokemon_texts", encoder, txt, c, l)
    c = l+1

qdrant_client.recreate_collection(
    collection_name="pokemon_images",
    vectors_config=models.VectorParams(
        size=1024,  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)
upload_images_to_qdrant(qdrant_client, "pokemon_images", "data/vector_pokemon.npy", labels)

qdrant_client.recreate_collection(
    collection_name="semantic_cache",
    vectors_config=models.VectorParams(
        size=768,  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)