<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual Domain-Specific Translation Model</title>
</head>
<body>
    <h1>Multilingual Domain-Specific Translation Model</h1>

    <h2>Overview</h2>
    <p>
        This project implements a <strong>multilingual machine translation system</strong> fine-tuned for translating domain-specific content, particularly related to <strong>health insurance and TPA (Third Party Administrator)</strong> across several languages, including <strong>English, Arabic, French, Romanian, Italian, Spanish, Russian, Turkish, and Greek</strong>.
    </p>
    <p>
        The project leverages <strong>Hugging Face's MarianMT pre-trained models</strong> and fine-tunes them on a small, domain-specific dataset. The system incorporates advanced NLP preprocessing techniques, language detection, and multiple evaluation metrics to ensure high-quality translations.
    </p>

    <hr>

    <h2>Features</h2>
    <ul>
        <li><strong>Multilingual Translation:</strong> Supports translation across several languages including:
            <ul>
                <li>English (en)</li>
                <li>Arabic (ar)</li>
                <li>French (fr)</li>
                <li>Romanian (ro)</li>
                <li>Italian (it)</li>
            </ul>
        </li>
        <li><strong>Domain-Specific Focus:</strong> Specializes in translations related to <strong>health insurance</strong> and <strong>TPA</strong> vocabulary and terminology.</li>
        <li><strong>Custom Language Preprocessing:</strong> Language-specific preprocessing steps for each language (e.g., handling accents in French, diacritics in Romanian, and special punctuation in Turkish).</li>
        <li><strong>Model Fine-Tuning:</strong> Fine-tuned MarianMT models for domain-specific translation tasks with support for custom tokenization, inference, and batching.</li>
        <li><strong>Language Detection:</strong> Detects the input language using <strong>FastText</strong> and dynamically loads the appropriate translation model.</li>
        <li><strong>Batch Translation with Beam Search:</strong> Translations are performed in batches with support for beam search to improve translation quality.</li>
        <li><strong>Evaluation Metrics:</strong> Multiple metrics to evaluate the quality of the translations:
            <ul>
                <li><strong>BLEU</strong></li>
                <li><strong>chrF</strong></li>
                <li><strong>BERTScore</strong></li>
            </ul>
        </li>
    </ul>

    <hr>

    <h2>Project Structure</h2>
    <pre>
.
├── data_prep.py            # Code for preparing and preprocessing the dataset for training
├── train.py                # Fine-tuning and training MarianMT models
├── inference.py            # Code for performing inference (translations) using the fine-tuned models
├── metrics.py              # Evaluation code to compute BLEU, chrF, and BERTScore
├── converter.py            # Convert data from tmx to csv
├── README.md               # Project documentation
└── .env                    # Environment variables (e.g., Hugging Face API keys)
    </pre>


  
    <hr>

    <h2>Installation from TestPyPI</h2>
    <p>You can install the package directly from TestPyPI:</p>
    <pre>
pip install -i https://test.pypi.org/simple/ lingowiz
    </pre>
    <p>For more information, you can view the package and its releases on TestPyPI: <a href="https://test.pypi.org/manage/project/lingowiz/releases/">Lingowiz on TestPyPI</a>.</p>

    <hr>

    <h2>Streamlit Interface and Google Colab Server</h2>
    <p>This project includes a <strong>Streamlit interface</strong> and is deployed using a <strong>Google Colab notebook as a server</strong>.</p>
    <p>To run the Streamlit interface, use the following command:</p>
    <pre>
streamlit run app.py
    </pre>
    <p>The Colab notebook is responsible for handling model inference as a server. You can use the notebook as a backend service for the translation model.</p>

    <hr>

    <h2>Usage</h2>
    <h3>1. Data Preprocessing</h3>
    <p>Prepare the data for training by using the <code>data_prep.py</code> script. This script handles data normalization, diacritics removal, and language-specific preprocessing.</p>

    <h3>2. Model Training</h3>
    <p>Train and fine-tune the MarianMT model using the <code>train.py</code> script.</p>
    <pre>
from train import training_pipeline

# Sample call to start the training process
training_pipeline(df, "general or special (domain specific)", "English", "Helsinki-NLP/opus-mt-en-ar", steps=1000, batch_size=32, lr=1e-5, epochs=3, warmup=500)
    </pre>

    <h3>3. Translation</h3>
    <p>Perform inference on a batch of texts using the <code>inference.py</code> script. The script automatically detects the input language and translates it into Arabic or the target language.</p>
    <pre>
from inference import inference

text = "Sample sentence for translation"
translated_text, language_name = inference(text)
print(f"Translated Text: {translated_text}")
print(f"Detected Language: {language_name}")
    </pre>

    <h3>4. Evaluation</h3>
    <p>Evaluate the performance of the translation model using the <code>metrics.py</code> script. This computes <strong>BLEU</strong>, <strong>chrF</strong>, and <strong>BERTScore</strong>.</p>
    <pre>
from metrics import evaluation

evaluation(df_test, model, tokenizer, "English")
    </pre>

    <hr>

    <h2>Evaluation Metrics</h2>
    <ul>
        <li><strong>BLEU:</strong> Measures how many n-grams in the prediction match the reference.</li>
        <li><strong>chrF:</strong> Computes character-level F-score between the prediction and reference.</li>
        <li><strong>BERTScore:</strong> Leverages BERT embeddings to measure the semantic similarity between prediction and reference.</li>
    </ul>

    <hr>

    <h2>Preprocessing Overview</h2>
    <p>Each language has a custom preprocessing pipeline tailored to its linguistic rules:</p>
    <ul>
        <li><strong>English:</strong> Handles contractions and lowercasing.</li>
        <li><strong>French:</strong> Normalizes accents and punctuation.</li>
        <li><strong>Romanian:</strong> Normalizes diacritics and cleans punctuation.</li>
        <li><strong>And more…</strong></li>
    </ul>

    ## TMX to CSV Converter

This project also includes a utility to convert TMX (Translation Memory eXchange) files to CSV format, which makes it easier to preprocess and use in training. Additionally, it supports decompressing `.tmx.gz` files for easier handling of compressed translation datasets.

### How to Use the Converter

1. **Decompress TMX.gz files:**
    The converter will automatically decompress TMX files if they are in `.gz` format.

2. **Convert TMX to CSV:**
    After decompression, the converter transforms the TMX data into a CSV file for easier preprocessing and integration with your training pipeline.

You can find the relevant code in the converter file located in the project structure.



    <hr>

## Data Constraints

The project faces some constraints in terms of data availability, particularly domain-specific sentences related to health insurance and TPA. To overcome this, the dataset was supplemented by generating sentences for specific terms, with around 200 sentences for each key term. This might affect the model's generalization ability, but it performs well given the current data limitations.

<hr>

    <h2>Future Improvements</h2>
    <ul>
        <li><strong>Expand dataset:</strong> Incorporate more domain-specific data to improve translation accuracy.</li>
        <li><strong>Additional languages:</strong> Extend support to more languages.</li>
        <li><strong>Model improvements:</strong> Explore using other pre-trained models for better accuracy in low-resource settings.</li>
        <li><strong>Deploy as an API:</strong> Package the model into a REST API for easier access and usage.</li &#8203;:contentReference[oaicite:0]{index=0}&#8203;
        <li>To further enhance the model's accuracy and generalization, more domain-specific data will be collected, and additional preprocessing techniques might be employed to ensure higher translation quality.</li>
