# Importing Librairies
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datasets import Dataset
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments,GenerationConfig # HF librairies for fine tuning
import pandas as pd # DataFrame manipulation (csv file)
import evaluate
import numpy as np
import torch

def translation(df_test, model, tokenizer, source_column, batch_size=32, max_length=128, num_beams=2):
    """
    Function to load a fine-tuned MarianMT model and perform translation on a DataFrame column with a progress bar.

    Parameters:
    - df_test: DataFrame containing the texts to be translated.
    - model: The fine-tuned MarianMT model.
    - tokenizer: The tokenizer for the model.
    - source_column: The name of the column in the DataFrame that contains the text in the source language.
    - batch_size: Number of texts to process in each batch (default is 32).
    - max_length: The maximum length of input sequences for translation (default is 128).
    - num_beams: The number of beams for beam search during inference (default is 5).

    Returns:
    - translated_texts: A list of translated texts.
    """

    # Detect if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the appropriate device
    model = model.to(device)

    # Initialize a list to store the translated texts
    translated_texts = []

    # Process the input data in batches
    for i in tqdm(range(0, len(df_test), batch_size), desc="Translating"):
        # Get a batch of texts
        batch_texts = df_test[source_column][i:i+batch_size].tolist()

        # Tokenize the input texts and move them to the same device as the model
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        # Perform translation (inference)
        translated_tokens = model.generate(**inputs, num_beams=num_beams)

        # Decode the generated tokens to human-readable text
        batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

        # Append the batch translations to the final list
        translated_texts.extend(batch_translations)

    return translated_texts


def metric_compute(predicted_texts, df_test, metric):
    """
    Function to compute evaluation metrics for comparing predicted outputs with reference data.

    Parameters:
    - predicted_texts: List or array of predicted outputs (e.g., from a model).
    - df_test: DataFrame containing the reference outputs in one of the columns (e.g., 'Arabic').
    - metric: The name of the evaluation metric to compute (e.g., 'bleu', 'bertscore', etc.).

    Returns:
    - metric: The computed metric score(s) and related statistics, as per the selected evaluation metric.
    """
    
    # Convert the list of predicted texts to a NumPy array and flatten it (to ensure proper shape)
    predictions = np.array(predicted_texts).flatten()

    # Extract the reference outputs from the specified DataFrame column
    reference = np.array(list(df_test["Arabic"]))

    # Optionally process the reference texts to remove language tags or extra characters
    # (e.g., removing the first 8 characters if they are tags like ">>fra<<")
    reference = [text[8:] for text in reference]

    # Convert the list of processed reference texts back to a NumPy array for further computation
    reference = np.array(reference)

    # Reshape the reference array to match the expected format (e.g., (num_samples, 1))
    reference = reference.reshape((len(reference), 1))

    # Load the specified evaluation metric from the Hugging Face `evaluate` library
    evaluator = evaluate.load(metric)

    # Compute the metric by comparing predictions to references
    # If the metric is 'bertscore', additional arguments (e.g., language) may be needed
    if metric == "bertscore":
        metric_result = evaluator.compute(predictions=predictions, references=reference, lang="en")
    else:
        metric_result = evaluator.compute(predictions=predictions, references=reference)

    # Return the computed metric result and related statistics
    return metric_result



def evaluation(df_test,model,tokenizer,source):
    """
    Evaluate the model predictions using BLEU, chrF, and BERTScore metrics.
    
    Args:
    - df_test: DataFrame containing the test data, including reference translations.
    
    This function prints the computed metrics for BLEU, chrF, and BERTScore.
    """
    
    # Generate predicted translations using the model
    predicted_texts = translation(df_test,model,tokenizer,source)

    # Compute the BLEU metric
    metric_bleu = metric_compute(predicted_texts, df_test, "bleu")

    # Compute the chrF metric
    metric_chrf = metric_compute(predicted_texts, df_test, "chrf")

    # Compute the BERTScore metric
    metric_bert = metric_compute(predicted_texts, df_test, "bertscore")

    # Print BLEU score
    print("\n--- BLEU Metric ---")
    print(f"BLEU Score: {metric_bleu['bleu']:.4f}")
    print(f"Precision: {metric_bleu['precisions']}")
    print(f"Brevity Penalty: {metric_bleu['brevity_penalty']:.4f}")
    
    # Print chrF metric
    print("\n--- chrF Metric ---")
    print(f"chrF Score: {metric_chrf['score']:.4f}")
    
    # Print BERTScore metric
    print("\n--- BERTScore Metric ---")
    print(metric_bert)
    # Optionally, print the predicted texts
    print("\n--- Sample Predicted Texts ---")
    for i, prediction in enumerate(predicted_texts[:5]):  # Print first 5 predicted texts as a sample
        print(f"Prediction {i+1}: {prediction}")
