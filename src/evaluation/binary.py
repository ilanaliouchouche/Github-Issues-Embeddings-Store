import os
import gc
import pandas as pd
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator

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
