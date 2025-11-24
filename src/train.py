import os
import json
from typing import Dict, Any
from collections import Counter

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from src.models.model_utils import load_tokenizer_and_model, tokenize_batch


SEED = 42
MODEL_NAME = "roberta-base"
PROCESSED_DATA_PATH = "data/processed"
LABEL2ID_PATH = "data/label2id.json"
ID2LABEL_PATH = "data/id2label.json"
OUTPUT_DIR = "models/cognitive_distortion_roberta"


def load_data_and_label_maps():
    """
    Load the processed dataset and label mappings saved by prepare_data.py
    """
    print(f"Loading processed dataset from: {PROCESSED_DATA_PATH}")
    dataset = load_from_disk(PROCESSED_DATA_PATH)

    with open(LABEL2ID_PATH, "r") as f:
        label2id = json.load(f)
    with open(ID2LABEL_PATH, "r") as f:
        id2label_str_keys = json.load(f)

    # Convert id2label keys back to int (JSON stores keys as strings)
    id2label = {int(k): v for k, v in id2label_str_keys.items()}

    print("Number of labels:", len(label2id))
    return dataset, label2id, id2label


def tokenize_datasets(dataset, tokenizer):
    """
    Apply the tokenizer to the train and validation splits.
    """

    def _tokenize_function(batch):
        return tokenize_batch(batch, tokenizer)

    tokenized = dataset.map(
        _tokenize_function,
        batched=True,
        remove_columns=["text", "dominant_distortion"],  # keep only model inputs + label
    )

    # Set format for PyTorch
    tokenized.set_format(type="torch")

    return tokenized


def compute_metrics(eval_pred) -> Dict[str, Any]:
    """
    Compute accuracy and F1 scores from model predictions.
    This will be used by the Hugging Face Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def get_class_weights_from_raw(train_split, num_labels: int):
    """
    Compute class weights from the *raw* train split (before tensor formatting).
    train_split is a datasets.Dataset where 'label' is a list of Python ints.

    weight_i = max_count / count_i
    """
    labels = train_split["label"]  # list of ints
    counts = Counter(labels)
    print("Label counts (raw train):", counts)

    max_count = max(counts.values())
    weights = [max_count / counts.get(i, 1) for i in range(num_labels)]
    print("Class weights:", weights)
    return weights


class WeightedTrainer(Trainer):
    """
    Custom Trainer that uses class-weighted CrossEntropyLoss.
    """

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            # store as tensor; will move to correct device in compute_loss
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Accept num_items_in_batch to be compatible with newer Trainer API.
        We don't actually need to use it; we just ignore it.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=weights)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        if return_outputs:
            return loss, outputs
        return loss


def main():
    set_seed(SEED)

    # 1) Load data + label mappings
    dataset, label2id, id2label = load_data_and_label_maps()

    # 2) Compute class weights from the *raw* train split (before tokenization)
    num_labels = len(label2id)
    class_weights = get_class_weights_from_raw(dataset["train"], num_labels=num_labels)

    # 3) Load tokenizer and model
    print(f"Loading model '{MODEL_NAME}' with {num_labels} labels...")
    tokenizer, model = load_tokenizer_and_model(
        model_name=MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 4) Tokenize dataset
    print("Tokenizing datasets...")
    tokenized = tokenize_datasets(dataset, tokenizer)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    # 5) Data collator (dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6) Training arguments (keep 4 epochs to give minorities a chance)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],  # no wandb or others by default
        seed=SEED,
    )

    # 7) Trainer (weighted)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8) Train
    print("\nStarting training with class-weighted loss...")
    trainer.train()
    print("\nTraining complete.")

    # 9) Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    # 10) Detailed classification report
    print("\nGenerating detailed classification report...")
    preds_output = trainer.predict(eval_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(
        classification_report(
            labels,
            preds,
            target_names=target_names,
            digits=4,
        )
    )

    # 11) Save model and tokenizer
    print(f"\nSaving model and tokenizer to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved.")


if __name__ == "__main__":
    main()
