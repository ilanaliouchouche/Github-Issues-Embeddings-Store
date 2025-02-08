from bs4 import BeautifulSoup
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers.utils import logging
import multiprocessing
import warnings

logging.set_verbosity_error()

warnings.filterwarnings("ignore")

SEED = 42

num_workers = multiprocessing.cpu_count()

dataset_dict = load_dataset("WhereIsAI/github-issue-similarity", "default")

print(dataset_dict)
dataset_dict = dataset_dict.rename_columns({"text1": "sentence1", "text2": "sentence2"})

binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=dataset_dict["valid"]["sentence1"],
    sentences2=dataset_dict["valid"]["sentence2"],
    labels=dataset_dict["valid"]["label"],
    name="git-issues",
    write_csv=False
)

def remove_html_tags(sample):
    sample["sentence1"] = BeautifulSoup(sample["sentence1"], "html.parser").get_text()
    sample["sentence2"] = BeautifulSoup(sample["sentence2"], "html.parser").get_text()
    return sample

dataset_dict = dataset_dict.map(remove_html_tags, num_proc=num_workers)

dataset_dict["train"] = dataset_dict["train"].filter(lambda x: x["sentence1"] != "" and x["sentence2"] != "")
dataset_dict["valid"] = dataset_dict["valid"].filter(lambda x: x["sentence1"] != "" and x["sentence2"] != "")
dataset_dict["test"] = dataset_dict["test"].filter(lambda x: x["sentence1"] != "" and x["sentence2"] != "")

def hpo_search_space(trial):
    return {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 8),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "scheduler_type": trial.suggest_categorical("scheduler_type", ["linear", "cosine", "cosine_with_restarts"]),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 5.0)
    }

def hpo_model_init(trial):
    return SentenceTransformer("NovaSearch/stella_en_400M_v5",
                               trust_remote_code=True)

def hpo_loss_init(model):
    return losses.OnlineContrastiveLoss(model)

def hpo_compute_objective(metrics):
    return metrics["eval_git-issues_cosine_mcc"]

args = SentenceTransformerTrainingArguments(
    output_dir=f"checkpoints/stella_onlinecontrastive",
    # fp16=True,
    bf16=True,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=10,
    run_name="stella-onlinecontrastive-hpo",
    seed=SEED,
    logging_dir="./runs/hpo-stella-onlinecontrastive",
    report_to=["tensorboard"],
    disable_tqdm=True
)

trainer = SentenceTransformerTrainer(
    model=None,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["valid"],
    evaluator=binary_acc_evaluator,
    model_init=hpo_model_init,
    loss=hpo_loss_init
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    hp_space=hpo_search_space,
    compute_objective=hpo_compute_objective,
    n_trials=5,
    backend="optuna"
)

print(best_trial)