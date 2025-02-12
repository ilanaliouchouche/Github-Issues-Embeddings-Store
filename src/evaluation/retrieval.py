from datasets import load_dataset
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize  # type: ignore
import nltk  # type: ignore
from sentence_transformers.util import cos_sim
import time
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import pickle
import os
import gc
import json

nltk.download("punkt_tab")

NUM_CORES = os.cpu_count()

CHECKPOINT_DIR = "./checkpoints"

MODELS_PATH = {
    model: os.path.join(CHECKPOINT_DIR, model, "best-model")
    for model in os.listdir(CHECKPOINT_DIR)
    if os.path.isdir(os.path.join(CHECKPOINT_DIR, model, "best-model"))
}

print(json.dumps(MODELS_PATH, indent=4))

ds = load_dataset("WhereIsAI/github-issue-similarity", "default")

ds["train"] = ds["train"].filter(lambda x: x["text1"] != "" and x["text2"] != "")
ds["valid"] = ds["valid"].filter(lambda x: x["text1"] != "" and x["text2"] != "")
ds["test"] = ds["test"].filter(lambda x: x["text1"] != "" and x["text2"] != "")

ds = ds.rename_columns({"text1": "sentence1", "text2": "sentence2"})

def remove_html_tags(sample):
    sample["sentence1"] = BeautifulSoup(sample["sentence1"], "html.parser").get_text().strip()
    sample["sentence2"] = BeautifulSoup(sample["sentence2"], "html.parser").get_text().strip()
    return sample

ds = ds.map(remove_html_tags, num_proc=NUM_CORES)

baseline_tfidf = TfidfVectorizer(tokenizer=word_tokenize, stop_words="english")

baseline_tfidf.fit(ds["train"]["sentence1"] + ds["train"]["sentence2"])


class RetrievalEvaluator:
    def __init__(self, models_path, baseline, dataset, save_path="rsrc", k_values=[1, 3, 5, 10]):
        self.models_path = models_path
        self.baseline = baseline
        self.dataset = dataset
        self.save_path = save_path
        self.k_values = k_values
        self.results = []
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def _compute_metrics(self, ranked_indices, relevant_idx):
        mrr = 0
        hits = {k: 0 for k in self.k_values}

        for i, idx in enumerate(ranked_indices):
            if idx == relevant_idx:
                mrr = 1 / (i + 1)
                for k in self.k_values:
                    if i < k:
                        hits[k] = 1
                break

        return mrr, hits

    def _evaluate_model(self,
                        model: SentenceTransformer):
        test_queries = self.dataset["test"]["sentence1"]
        test_docs = self.dataset["test"]["sentence2"]
        relevant_indices = np.arange(len(test_queries))

        start_encode = time.time()
        embeddings = model.encode(test_queries + test_docs, convert_to_tensor=True, show_progress_bar=True, batch_size=16)
        time_per_sample = (time.time() - start_encode)/len(test_queries)
        embeddings_queries = embeddings[:len(test_queries)]
        embeddings_docs = embeddings[len(test_queries):]
        
        similarity_matrix = cos_sim(embeddings_queries, embeddings_docs).cpu().numpy()
        rankings = np.argsort(-similarity_matrix, axis=1)

        mrr_scores = []
        hits_scores = {k: [] for k in self.k_values}

        for i, ranked_indices in enumerate(rankings):
            mrr, hits = self._compute_metrics(ranked_indices, relevant_indices[i])
            mrr_scores.append(mrr)
            for k in self.k_values:
                hits_scores[k].append(hits[k])

        avg_mrr = np.mean(mrr_scores)
        avg_hits = {k: np.mean(hits_scores[k]) for k in self.k_values}

        return avg_mrr, avg_hits, time_per_sample

    def evaluate(self):
        for model_name, model_path in self.models_path.items():
            print(f"Evaluating model: {model_name}...")
            model = SentenceTransformer(model_path, trust_remote_code=True, device=self.device)
            avg_mrr, avg_hits, time_per_sample = self._evaluate_model(model)
            self.results.append({
                "model": model_name,
                "mrr": avg_mrr,
                **{f"hits@{k}": avg_hits[k] for k in self.k_values},
                "time": time_per_sample
            })
            del model
            gc.collect()
            torch.cuda.empty_cache()

        print("Evaluating baseline TF-IDF...")
        embeddings = self.baseline.transform(self.dataset["test"]["sentence1"] + self.dataset["test"]["sentence2"])
        embeddings_queries = embeddings[:len(self.dataset["test"]["sentence1"])]
        embeddings_docs = embeddings[len(self.dataset["test"]["sentence1"]):]

        similarity_matrix = (embeddings_queries @ embeddings_docs.T).toarray()
        rankings = np.argsort(-similarity_matrix, axis=1)

        mrr_scores = []
        hits_scores = {k: [] for k in self.k_values}
        times_per_sample = []

        for i, ranked_indices in enumerate(rankings):
            start_eval = time.time()
            mrr, hits = self._compute_metrics(ranked_indices, i)
            mrr_scores.append(mrr)
            for k in self.k_values:
                hits_scores[k].append(hits[k])
            times_per_sample.append(time.time() - start_eval)

        avg_mrr = np.mean(mrr_scores)
        avg_hits = {k: np.mean(hits_scores[k]) for k in self.k_values}
        time_per_sample = np.mean(times_per_sample)

        self.results.append({
            "model": "TF-IDF Baseline",
            "mrr": avg_mrr,
            **{f"hits@{k}": avg_hits[k] for k in self.k_values},
            "time": time_per_sample
        })

        return self._save_results()

    def _save_results(self):
        results_df = pd.DataFrame(self.results)
        os.makedirs(self.save_path, exist_ok=True)
        filename = f"retrieval_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        filename = os.path.join(self.save_path, filename)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return results_df

retr_eval = RetrievalEvaluator(MODELS_PATH, baseline_tfidf, ds, k_values=[1, 5, 10, 20])

print(retr_eval.evaluate())

print("Evaluation done!")
