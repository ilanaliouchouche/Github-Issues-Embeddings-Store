from bs4 import BeautifulSoup
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import multiprocessing

num_workers = multiprocessing.cpu_count()

dataset_dict = load_dataset("WhereIsAI/github-issue-similarity", "default")
dataset_dict = dataset_dict.rename_columns({"text1": "sentence1", "text2": "sentence2"})

binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=dataset_dict["valid"]["sentence1"],
    sentences2=dataset_dict["valid"]["sentence2"],
    labels=dataset_dict["valid"]["label"],
    name="git-issues"
)

def remove_html_tags(sample):
    sample["sentence1"] = BeautifulSoup(sample["text1"], "html.parser").get_text()
    sample["sentence2"] = BeautifulSoup(sample["text2"], "html.parser").get_text()
    return sample

dataset_dict = dataset_dict.map(remove_html_tags, num_proc=num_workers//2)
trainer = SentenceTransformerTrainer()

def hpo_search_space(trial):
    return {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 8),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.1, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "scheduler_type": trial.suggest_categorical("scheduler_type", ["linear", "cosine", "cosine_with_restarts"]),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 1.0)
    }

def hpo_model_init(trial):
    return SentenceTransformer("NovaSearch/stella_en_400M_v5")

def hpo_loss_init(model):
    return losses.ContrastiveLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=f"checkpoints/stella_contrastive",
    fp16=True,
    bf16=True,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    seed=42,
    logging_dir="runs",
    run_name="stella-contrastive",
)


#### TODO TODO TODO TODO TODO TODO TODO TODO
