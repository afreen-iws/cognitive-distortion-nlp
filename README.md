🧠 **Cognitive Distortion Detection using NLP**

**Detecting Biased Reasoning in Language using Transformer-based Models**

**Live Demo (Streamlit App)**: https://cognitive-distortion-nlp.streamlit.app/  

**Hugging Face Model**: Afreenfath/cognitive-distortion-roberta

Overview

This project develops an NLP-based system that detects cognitive distortions—thinking patterns linked to biased reasoning, anxiety, depression, and negative self-evaluation.

Unlike demographic/algorithmic bias detection, this project focuses on psychological bias in language, such as:

| Distortion Type    | Example                                           |
| ------------------ | ------------------------------------------------- |
| Overgeneralization | *“I failed once, so I’ll never succeed in life.”* |
| Mind Reading       | *“They didn’t text back — they must hate me.”*    |
| Personalization    | *“It rained because I planned a picnic.”*         |
| Catastrophizing    | *“This is a total disaster.”*                     |
| Should Statements  | *“I must be perfect all the time.”*               |

**Project Goals**

✔ Classify user text into 11 cognitive distortion categories
✔ Fine-tune RoBERTa-base on a public mental health dataset
✔ Deploy an interactive Streamlit App
✔ Include probability-based outputs, and heuristic keyword highlighting
✔ Build a clean, modular, production-style GitHub repository

**Dataset**

**Source**: psytechlab/cognitive_distortions_dataset_ru (Hugging Face)

| Feature | Description                                                                |
| ------- | -------------------------------------------------------------------------- |
| text    | User-generated statements from forums, counseling, mental health platforms |
| label   | One of 11 cognitive distortion types                                       |

**Model: RoBERTa Fine-Tuning for Text Classification**

| Config           | Value                                    |
| ---------------- | ---------------------------------------- |
| Base model       | roberta-base                             |
| Task             | Multi-class classification (11 labels)   |
| Loss function    | Weighted Cross Entropy (class imbalance) |
| Token max length | 256                                      |
| Optimizer        | AdamW                                    |
| Training epochs  | 4                                        |

Model was fine-tuned with class weights, due to imbalance in categories like Should statements and Mental Filter.

**Best-performing labels**:
✔ No Distortion
✔ Mind Reading
✔ Magnification

**Challenging labels**:
Mental Filter, Labeling, Emotional Reasoning (data scarcity)

**How to Run the Project Locally**
1️⃣ Clone the Repo
git clone https://github.com/YourRepo/cognitive-distortion-nlp.git
cd cognitive-distortion-nlp

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (Optional)
python src/train.py

4️⃣ Run the Streamlit App
streamlit run streamlit_app.py

**Live Streamlit Demo Features**
| Feature                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| Text input               | Users can paste any sentence or paragraph          |
| Predicted label          | Model returns predicted cognitive distortion       |
| Confidence scores        | Top-3 probabilities shown                          |
| Heuristic highlighting   | Keywords suggesting cognitive bias are highlighted |
| Educational descriptions | Shows explanation of distortion category           |
