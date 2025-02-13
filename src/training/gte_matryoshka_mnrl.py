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

dataset_dict["train"] = dataset_dict["train"].filter(lambda x: x["text1"] != "" and x["text2"] != "")
dataset_dict["valid"] = dataset_dict["valid"].filter(lambda x: x["text1"] != "" and x["text2"] != "")
dataset_dict["test"] = dataset_dict["test"].filter(lambda x: x["text1"] != "" and x["text2"] != "")

# FOR DEBUGGING
# dataset_dict["train"] = dataset_dict["train"].select(range(32))
# dataset_dict["valid"] = dataset_dict["valid"].select(range(32))
# dataset_dict["test"] = dataset_dict["test"].select(range(32))

def remove_html_tags(sample):
    sample["text1"] = BeautifulSoup(sample["text1"], "html.parser").get_text()
    sample["text2"] = BeautifulSoup(sample["text2"], "html.parser").get_text()
    return sample

dataset_dict = dataset_dict.map(remove_html_tags, num_proc=num_workers)

dataset_dict = dataset_dict.rename_columns({"text1": "anchor", "text2": "positive"})
dataset_dict["train"] = dataset_dict["train"].filter(lambda x: x["label"] == 1)
dataset_dict["train"] = dataset_dict["train"].remove_columns("label")

print(dataset_dict)

binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=dataset_dict["valid"]["anchor"],
    sentences2=dataset_dict["valid"]["positive"],
    labels=dataset_dict["valid"]["label"],
    name="git-issues",
    write_csv=False
)

model = SentenceTransformer("prdev/mini-gte",
                            trust_remote_code=True)

loss = losses.MultipleNegativesRankingLoss(model)
loss = losses.MatryoshkaLoss(model, loss, [768, 512, 128, 64])

args = SentenceTransformerTrainingArguments(
    output_dir=f"checkpoints/gte-mat-mnrl",
    num_train_epochs=12,
    per_device_train_batch_size=128,
    learning_rate=4e-6,
    warmup_ratio=0.15,
    weight_decay=0.2,
    lr_scheduler_type='cosine',
    max_grad_norm=4.7,
    # fp16=True,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    run_name="gte-mat-mnrl",
    seed=SEED,
    logging_dir="./runs/gte-mat-mnrl",
    report_to=["tensorboard"],
    disable_tqdm=False,
    load_best_model_at_end=True
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["valid"],
    evaluator=binary_acc_evaluator,
    loss=loss
)

trainer.train()

print(trainer.evaluate(dataset_dict["test"], binary_acc_evaluator))

trainer.save_model(f"checkpoints/gte-mat-mnrl/best-model")
