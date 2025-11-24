import os
import json
from datasets import load_dataset, DatasetDict

SEED = 42
DATASET_NAME = "psytechlab/cognitive_distortions_dataset_ru"


def load_and_prepare_dataset():
    """
    1. Load the cognitive distortions dataset from Hugging Face.
    2. Create a 'text' column from 'patient_question'.
    3. Build label2id and id2label mappings from 'dominant_distortion'.
    4. Add a numeric 'label' column.
    5. Split into train/validation sets.
    6. Save processed dataset and mappings to disk.
    """
    print(f"Loading dataset: {DATASET_NAME}")
    raw_dataset = load_dataset(DATASET_NAME)

    # Create a unified 'text' column
    def add_text_column(example):
        example["text"] = example["patient_question"]
        return example

    dataset = raw_dataset.map(add_text_column)

    # Build label mappings from dominant_distortion in the train split
    unique_labels = sorted(list(set(dataset["train"]["dominant_distortion"])))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print("Found labels:")
    for label, idx in label2id.items():
        print(f"  {idx}: {label}")

    # Add numeric 'label' column using label2id
    def add_numeric_label(example):
        example["label"] = label2id[example["dominant_distortion"]]
        return example

    dataset = dataset.map(add_numeric_label)

    # Keep only the columns we care about for training
    # We'll keep: text (input), label (numeric), dominant_distortion (original label string)
    keep_cols = ["text", "label", "dominant_distortion"]
    for split in dataset.keys():
        cols_to_remove = [c for c in dataset[split].column_names if c not in keep_cols]
        dataset[split] = dataset[split].remove_columns(cols_to_remove)

    # Create train/validation split from the train split
    # NOTE: no stratification to avoid ClassLabel type issues
    print("\nCreating train/validation split (80/20)...")
    train_valid = dataset["train"].train_test_split(
        test_size=0.2,
        seed=SEED,
    )

    processed = DatasetDict(
        {
            "train": train_valid["train"],
            "validation": train_valid["test"],
        }
    )

    # Make sure output folder exists
    os.makedirs("data", exist_ok=True)
    processed_path = os.path.join("data", "processed")
    print(f"\nSaving processed dataset to: {processed_path}")
    processed.save_to_disk(processed_path)

    # Save label mappings
    with open(os.path.join("data", "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    with open(os.path.join("data", "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)

    print("\nSaved label2id and id2label to data/label2id.json and data/id2label.json")


def main():
    load_and_prepare_dataset()


if __name__ == "__main__":
    main()
