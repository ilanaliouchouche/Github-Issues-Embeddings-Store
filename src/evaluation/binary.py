import os
import gc
import pandas as pd
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from bs4 import BeautifulSoup
from datasets import load_dataset
import json

NUM_CORES = os.cpu_count()

CHECKPOINT_DIR = "./checkpoints"

MODELS_PATH = {
    model: os.path.join(CHECKPOINT_DIR, model, "best-model")
    for model in os.listdir(CHECKPOINT_DIR)
    if os.path.isdir(os.path.join(CHECKPOINT_DIR, model, "best-model"))
}

print(json.dumps(MODELS_PATH, indent=4))

ds = load_dataset("WhereIsAI/github-issue-similarity", "default")

ds["test"] = ds["test"].filter(lambda x: x["text1"] != "" and x["text2"] != "")

ds = ds.rename_columns({"text1": "sentence1", "text2": "sentence2"})

def remove_html_tags(sample):
    sample["sentence1"] = BeautifulSoup(sample["sentence1"], "html.parser").get_text().strip()
    sample["sentence2"] = BeautifulSoup(sample["sentence2"], "html.parser").get_text().strip()
    return sample

ds["test"] = ds["test"].map(remove_html_tags, num_proc=NUM_CORES)

class BinaryRetrievalEvaluator:
    def __init__(self, models_path, dataset, save_path="rsrc"):
        self.models_path = models_path
        self.dataset = dataset
        self.save_path = save_path
        self.results = []
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def _evaluate_model(self, model: SentenceTransformer, model_name):
        evaluator = BinaryClassificationEvaluator(
            sentences1=self.dataset["test"]["sentence1"],
            sentences2=self.dataset["test"]["sentence2"],
            labels=self.dataset["test"]["label"],
            name=model_name,
            write_csv=False
        )
        metrics = evaluator(model)

        formatted_metrics = {"model": model_name}
        for key, value in metrics.items():
            formatted_metrics[key.split("_")[-1]] = value
        return formatted_metrics

    def evaluate(self):
        for model_name, model_path in self.models_path.items():
            print(f"Evaluating model: {model_name}...")
            model = SentenceTransformer(model_path, trust_remote_code=True, device=self.device)
            metrics = self._evaluate_model(model, model_name)
            self.results.append(metrics)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        return self._save_results()

    def _save_results(self):
        os.makedirs(self.save_path, exist_ok=True)
        results_df = pd.DataFrame(self.results)
        filename = os.path.join(self.save_path, f"binary_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return results_df

evaluator = BinaryRetrievalEvaluator(MODELS_PATH, ds)

print(evaluator.evaluate())

print("Evaluation done!")
