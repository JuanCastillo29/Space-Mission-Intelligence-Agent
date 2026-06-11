import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

class BGEReranker:
    def __init__(self, model_name:str = MODEL_NAME, device :str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query, documents, batch_size = 16):
        scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:min(i + batch_size, len(documents))]

            pairs = [[query, doc] for doc in batch_docs]

            inputs  = self.tokenizer(

            )