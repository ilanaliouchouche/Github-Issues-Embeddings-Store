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

# FOR DEBUGGING
# dataset_dict["train"] = dataset_dict["train"].select(range(8))
# dataset_dict["valid"] = dataset_dict["valid"].select(range(8))
# dataset_dict["test"] = dataset_dict["test"].select(range(8))

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

print(dataset_dict)

model = SentenceTransformer("intfloat/multilingual-e5-small",
                            trust_remote_code=True)

loss = losses.ContrastiveLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=f"checkpoints/e5-contrastive",
    num_train_epochs=3,
    per_device_train_batch_size=32,
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
    run_name="e5-contrastive",
    seed=SEED,
    logging_dir="./runs/e5-contrastive",
    report_to=["tensorboard"],
    disable_tqdm=False,
    load_best_model_at_end=True
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["valid"],
    evaluator=binary_acc_evaluator
)

trainer.train()

print(trainer.evaluate(dataset_dict["test"], binary_acc_evaluator))

trainer.save_model(f"checkpoints/e5-contrastive/best-model")
