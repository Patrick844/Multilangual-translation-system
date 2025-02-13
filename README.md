
# Multilingual Domain-Specific Translation Model

## Overview

This project implements a **multilingual machine translation system** fine-tuned for translating domain-specific content, particularly related to **health insurance and TPA (Third Party Administrator)** across several languages, including **English, Arabic, French, Romanian, Italian, Spanish, Russian, Turkish, and Greek**.

The project leverages **Hugging Face's MarianMT pre-trained models** and fine-tunes them on a small, domain-specific dataset. The system incorporates advanced NLP preprocessing techniques, language detection, and multiple evaluation metrics to ensure high-quality translations.

---

## Challenges
In building the Multilingual Translation System, the biggest challenge I faced was the lack of domain-specific data, particularly in health insurance. I started with only a few key words and manually generated sentences to create a small dataset of 1000 samples. This limited data led me to freeze the final two encoder layers of the MarianMT model to preserve the underlying structure and focus fine-tuning on the specialized domain. The decision to freeze the last 2 encoder layers allowed me to retain generalization while improving performance on specialized health insurance terms.

Additionally, resource limitations made training from scratch impractical, so I leveraged Hugging Face's MarianMT model to save computational costs and ensure scalability. Balancing the need for fine-tuning with keeping the model general enough to handle unseen cases was key, alongside experimenting with different hyperparameters  like Learning rate, Epoch, and Warmup steps to avoid overfitting while optimizing translation accuracy.

---

![Fine Tuning](fr-ar-metrics.png)
*Figure 1: Fine Tuning Helsinki-NLP/opus-mt-fr-ar .*


## Features

- **Multilingual Translation:** Supports translation across several languages including:
  - English (en)
  - Arabic (ar)
  - French (fr)
  - Romanian (ro)
  - Italian (it)


- **Domain-Specific Focus:** Specializes in translations related to **health insurance** and **TPA** vocabulary and terminology.
- **Custom Language Preprocessing:** Language-specific preprocessing steps for each language (e.g., handling accents in French, diacritics in Romanian, and special punctuation in Turkish).
- **Model Fine-Tuning:** Fine-tuned MarianMT models for domain-specific translation tasks with support for custom tokenization, inference, and batching.
- **Language Detection:** Detects the input language using **FastText** and dynamically loads the appropriate translation model.
- **Batch Translation with Beam Search:** Translations are performed in batches with support for beam search to improve translation quality.
- **Evaluation Metrics:** Multiple metrics to evaluate the quality of the translations:
  - **BLEU**
  - **chrF**
  - **BERTScore**

---

## Project Structure

- `data_prep.py` - Code for preparing and preprocessing the dataset for training
- `train.py` - Fine-tuning and training MarianMT models
- `inference.py` - Code for performing inference (translations) using the fine-tuned models
- `metrics.py` - Evaluation code to compute BLEU, chrF, and BERTScore
- `converter.py` - Convert data from TMX to CSV
- `README.md` - Project documentation
- `.env` - Environment variables (e.g., Hugging Face API keys)


  
## Installation from TestPyPI

You can install the package directly from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ lingowiz
```

For more information, you can view the package and its releases on TestPyPI: [Lingowiz on TestPyPI](https://test.pypi.org/manage/project/lingowiz/releases/).

---


## Streamlit Interface and Google Colab Server

This project includes a **Streamlit interface** and is deployed using a **Google Colab notebook as a server**.

To run the Streamlit interface, use the following command:

```bash
streamlit run app.py
```

The Colab notebook is responsible for handling model inference as a server. You can use the notebook as a backend service for the translation model.


   ---

## Usage

### 1. Data Preprocessing
Prepare the data for training by using the `data_prep.py` script. This script handles data normalization, diacritics removal, and language-specific preprocessing.

### 2. Model Training
Train and fine-tune the MarianMT model using the `train.py` script.

```python
from train import training_pipeline
```

# Sample call to start the training process
```python
training_pipeline(df, "general or special (domain specific)", "English", "Helsinki-NLP/opus-mt-en-ar", steps=1000, batch_size=32, lr=1e-5, epochs=3, warmup=500)
```

### 3. Translation
Perform inference on a batch of texts using the `inference.py` script. The script automatically detects the input language and translates it into Arabic or the target language.

```python
from inference import inference

text = "Sample sentence for translation"
translated_text, language_name = inference(text)
print(f"Translated Text: {translated_text}")
print(f"Detected Language: {language_name}")
```
  ### 4. Evaluation
Evaluate the performance of the translation model using the `metrics.py` script. This computes **BLEU**, **chrF**, and **BERTScore**.

```python
from metrics import evaluation

evaluation(df_test, model, tokenizer, "English")
```

   ## Evaluation Metrics

- **BLEU:** Measures how many n-grams in the prediction match the reference.
- **chrF:** Computes character-level F-score between the prediction and reference.
- **BERTScore:** Leverages BERT embeddings to measure the semantic similarity between prediction and reference.

---

## Preprocessing Overview

Each language has a custom preprocessing pipeline tailored to its linguistic rules:

- **English:** Handles contractions and lowercasing.
- **French:** Normalizes accents and punctuation.
- **Romanian:** Normalizes diacritics and cleans punctuation.
- **And more...**

## TMX to CSV Converter

This project also includes a utility to convert TMX (Translation Memory eXchange) files to CSV format, which makes it easier to preprocess and use in training. Additionally, it supports decompressing `.tmx.gz` files for easier handling of compressed translation datasets.

### How to Use the Converter

1. **Decompress TMX.gz files:**  
   The converter will automatically decompress TMX files if they are in `.gz` format.

2. **Convert TMX to CSV:**  
   After decompression, the converter transforms the TMX data into a CSV file for easier preprocessing and integration with your training pipeline.

You can find the relevant code in the converter file located in the project structure.

---

## Data Constraints

The project faces some constraints in terms of data availability, particularly domain-specific sentences related to health insurance and TPA. To overcome this, the dataset was supplemented by generating sentences for specific terms, with around 200 sentences for each key term. This might affect the model's generalization ability, but it performs well given the current data limitations.

---

## Difficulties Faced and How They Were Solved

### 1. **Data Quality and Availability**
   - **Challenge**: The initial dataset provided was limited in size and quality, especially in terms of domain-specific terminology in healthcare and TPA.
   - **Solution**: I used a combination of **GPT-generated sentences** and **public healthcare datasets (like OPUS)** to augment the dataset. Preprocessing steps such as spell checking, tokenization, and segmentation were applied to improve data quality.

### 2. **Model Overfitting**
   - **Challenge**: The model initially overfit to the small dataset, especially on specialized terms, which resulted in poor generalization.
   - **Solution**: I mitigated this by freezing the **last few layers of the encoder, decoder and embeddings**, which allowed the model to retain its general knowledge of the source languages while specializing only in domain-specific terminology. Additionally, careful tuning of **learning rate** and **early stopping** helped combat overfitting.

### 3. **Evaluation Metrics and Logging**
   - **Challenge**: Incorporating and tracking multiple evaluation metrics like BLEU, chrF, and BERTScore during training was complex.
   - **Solution**: I implemented **MLflow** to log these metrics automatically at the end of each epoch, allowing for easy tracking and comparison of results across experiments.

### 4. **Handling Low-Quality Translations**
   - **Challenge**: Poor-quality translations in the dataset impacted overall model performance.
   - **Solution**: I applied various **post-processing steps** such as spell checking and sentence segmentation. Additionally, I'm working on a future improvement to implement a **dynamic loss function** that can adapt based on data quality and prioritize learning from more difficult samples.

## Project Structure

- `data_prep.py` - Code for preparing and preprocessing the dataset for training.
- `train.py` - Fine-tuning and training MarianMT models.
- `inference.py` - Code for performing inference (translations) using the fine-tuned models.
- `metrics.py` - Evaluation code to compute BLEU, chrF, and BERTScore.
- `converter.py` - Convert data from TMX to CSV.
- `inference_preprocessing.py` - Handles preprocessing for inference data.
- `README.md` - Project documentation.
- `.env` - Environment variables (e.g., Hugging Face API keys).
- `utils.py` - Contains dictionary and utilities.

## MLflow Integration

   - **Extensive logging**: All components of the project are logged using MLflow, including:
     - **Training and evaluation metrics** such as loss, BLEU, chrF, and BERTScore.
     - **Train datasets**: The datasets used during training are logged for reproducibility and analysis.
     - **Inference data with custom ratings**: The inference results are logged along with custom ratings for translation quality, providing insights into model performance on different types of data.
     - **Custom metrics**: Additional metrics are logged to analyze the model's performance and behavior in specific scenarios, with the goal of optimizing future fine-tuning and improvements.

## Future Improvements

- **Dynamic Loss Function**: The proposed custom loss function can be explored to enhance model training, particularly for low-quality or noisy datasets. By dynamically adjusting the loss based on sentence quality, the model can focus more on challenging data.
- **Additional Languages**: Future iterations of the project could expand support to more languages beyond the current six.
- **Improved Evaluation Techniques**: Exploring new evaluation metrics that may provide more granular insights into translation quality for medical and TPA-specific content.