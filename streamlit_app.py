import json
from typing import List, Tuple

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Config ----------

MODEL_DIR = "models/cognitive_distortion_roberta"
LABEL2ID_PATH = "data/label2id.json"
ID2LABEL_PATH = "data/id2label.json"
MAX_LENGTH = 256


# ---------- Helper functions ----------

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the fine-tuned model and tokenizer from disk.
    Cached so Streamlit only loads them once.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    model.to("cpu")  # Streamlit Cloud usually runs on CPU
    return tokenizer, model


@st.cache_resource
def load_label_maps():
    """
    Load label2id and id2label mappings saved during preprocessing.
    """
    with open(LABEL2ID_PATH, "r") as f:
        label2id = json.load(f)
    with open(ID2LABEL_PATH, "r") as f:
        id2label_str = json.load(f)
    id2label = {int(k): v for k, v in id2label_str.items()}
    return label2id, id2label


def get_label_descriptions():
    """
    Human-readable explanations for each cognitive distortion label.
    """
    return {
        "All-or-nothing thinking": "Seeing things in black-and-white categories (e.g., 'always', 'never', 'completely').",
        "Emotional Reasoning": "Assuming that because you feel something, it must be true (e.g., 'I feel useless, so I am useless').",
        "Fortune-telling": "Predicting the future negatively without realistic evidence (e.g., 'This will definitely fail').",
        "Labeling": "Assigning a global, negative label to yourself or others (e.g., 'I am a loser').",
        "Magnification": "Blowing problems or mistakes out of proportion; catastrophizing.",
        "Mental filter": "Focusing only on negatives and ignoring positives; filtering out good evidence.",
        "Mind Reading": "Assuming you know what others are thinking without proof (e.g., 'They all think I’m stupid').",
        "No Distortion": "No strong cognitive distortion detected; the reasoning seems relatively balanced.",
        "Overgeneralization": "Drawing sweeping conclusions from a single event (e.g., 'I failed once, I’ll always fail').",
        "Personalization": "Blaming yourself for events outside your control; taking things too personally.",
        "Should statements": "Using rigid 'should', 'must', or 'ought' rules about yourself or others.",
    }


def heuristic_highlight(text: str, predicted_label: str) -> str:
    """
    Very simple keyword-based highlighting.
    This is NOT using model internals, just pattern-based hints.

    We wrap matched words in <mark> tags so Streamlit can show them with HTML.
    """
    lower = text.lower()

    keywords = {
        "All-or-nothing thinking": ["always", "never", "completely", "totally", "every time", "nothing"],
        "Fortune-telling": ["will definitely", "going to fail", "it will be a disaster", "i know it will"],
        "Magnification": ["disaster", "ruined", "terrible", "horrible", "the worst", "unbearable"],
        "Should statements": ["should", "must", "have to", "ought to"],
        "Overgeneralization": ["everyone", "no one", "everything", "nothing ever", "i always", "i never"],
        "Personalization": ["my fault", "because of me", "all on me"],
        "Mind Reading": ["they think", "everyone thinks", "people think", "they must think"],
        "Mental filter": ["only bad", "nothing good", "ignore the good"],
        "Emotional Reasoning": ["i feel like", "i feel that"],
    }

    phrases = keywords.get(predicted_label, [])
    if not phrases:
        return text

    highlighted = text
    offset = 0  # track index shifts as we insert <mark> tags

    for phrase in phrases:
        phrase_lower = phrase.lower()
        idx = lower.find(phrase_lower)
        if idx != -1:
            start = idx + offset
            end = start + len(phrase)
            original_sub = highlighted[start:end]
            marked = f"<mark>{original_sub}</mark>"

            highlighted = highlighted[:start] + marked + highlighted[end:]
            offset += len(marked) - len(original_sub)
            lower = highlighted.lower()

    return highlighted


def predict_distortion(text: str) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Run the model on a single input text and return:
    - predicted label
    - list of (label, probability) for the top 3 classes
    """
    tokenizer, model = load_model_and_tokenizer()
    _, id2label = load_label_maps()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    pred_label = id2label[pred_id]

    # Top-3 labels
    top_probs, top_ids = torch.topk(probs, k=3)
    top_classes = []
    for p, idx in zip(top_probs, top_ids):
        label = id2label[int(idx.item())]
        top_classes.append((label, float(p.item())))

    return pred_label, top_classes


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(
        page_title="Cognitive Distortion Detector",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Detecting Biased Reasoning in Language")
    st.markdown(
        """
        This app uses a fine-tuned **RoBERTa** model to classify short texts into 
        cognitive distortion types (e.g., catastrophizing, overgeneralization) or **No Distortion**.
        
        Paste a sentence or short paragraph below to analyse its reasoning pattern.
        """
    )

    with st.sidebar:
        st.header("About this project")
        st.write(
            """
            **Goal:** Detect cognitive distortions in everyday language using NLP.
            
            **Model:** RoBERTa-base fine-tuned on a public cognitive distortion dataset.
            
            **Note:** This tool is **not** a clinical diagnostic system. It is for 
            educational and research purposes only.
            """
        )

    text = st.text_area(
        "Enter a sentence or short paragraph:",
        height=180,
        placeholder="Example: \"I failed this exam, so I'm never going to succeed at anything.\"",
    )

    if st.button("Analyse Text"):
        if not text.strip():
            st.warning("Please enter some text first.")
            return

        with st.spinner("Analysing..."):
            pred_label, top_classes = predict_distortion(text)
            label_descriptions = get_label_descriptions()
            explanation = label_descriptions.get(
                pred_label,
                "No description available for this label.",
            )
            highlighted = heuristic_highlight(text, pred_label)

        st.subheader("Prediction")
        st.markdown(f"**Predicted Category:** `{pred_label}`")
        st.markdown(f"**Explanation:** {explanation}")

        st.subheader("Class Probabilities (Top 3)")
        for label, prob in top_classes:
            st.write(f"- **{label}**: {prob:.3f}")

        st.subheader("Highlighted Text (heuristic)")
        st.markdown(
            """
            The highlighted words/phrases are guessed **heuristically** based on the 
            predicted distortion type (not full attention interpretability).
            """,
            help="This is a simple keyword-based heuristic, not a full explanation of the model's internals.",
        )
        st.markdown(highlighted, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
