"""
Function: translation

This function generates translations for a specified column in a DataFrame 
using a fine-tuned MarianMT model.

Parameters
----------
df_test : pandas.DataFrame
    The DataFrame containing the texts to be translated.
model : transformers.MarianMTModel
    The fine-tuned MarianMT model used for translation.
tokenizer : transformers.MarianTokenizer
    The tokenizer corresponding to the MarianMT model.
source_column : str
    The name of the column in the DataFrame that contains the source texts.
batch_size : int, optional
    The number of texts to process in each batch (default is 8).
max_length : int, optional
    The maximum length of input sequences for translation (default is 128).
num_beams : int, optional
    The number of beams for beam search during inference (default is 2).

Returns
-------
list
    A list of translated texts.

Example
-------
.. code-block:: python

    from transformers import MarianTokenizer, MarianMTModel
    from lingowiz.metrics import translation
    import pandas as pd

    # Load model and tokenizer
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

    # Prepare test data
    df_test = pd.DataFrame({"English": ["Hello world!", "How are you?"]})

    # Translate
    translations = translation(
        df_test=df_test,
        model=model,
        tokenizer=tokenizer,
        source_column="English"
    )
    print(translations)
"""
# Importing Libraries
import evaluate
import numpy as np
import torch
import mlflow


def translation(df_test,
                model,
                tokenizer,
                source_column,
                batch_size=8,
                max_length=128,
                num_beams=2):
    """
    Function to load a fine-tuned MarianMT model and perform translation on
    a DataFrame column using Swifter for parallel processing.

    Parameters
    - df_test: DataFrame containing the texts to be translated.
    - model: The fine-tuned MarianMT model.
    - tokenizer: The tokenizer for the model.
    - source_column: The name of the column in the DataFrame that contains the text in the source language.
    - batch_size: Number of texts to process in each batch (default is 8).
    - max_length: The maximum length of input sequences for translation (default is 128).
    - num_beams: The number of beams for beam search during inference (default is 2).

    Returns:
    - translated_texts: A list of translated texts.
    """

    torch.cuda.empty_cache()
    # Detect if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model = model.to(device)

    # Define a translation function for Swifter
    def translate_text(text):
        # Tokenize the input text and move to the appropriate device
        inputs = tokenizer(text,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=max_length).to(device)

        # Perform translation (inference)
        translated_tokens = model.generate(**inputs, num_beams=num_beams)

        # Decode the generated tokens to human-readable text
        translation = tokenizer.decode(translated_tokens[0],
                                       skip_special_tokens=True)
        return translation

    # Apply Swifter to process the source_column with parallelization
    translated_texts = df_test[source_column].swifter.apply(translate_text)

    return translated_texts.tolist()


def metric_compute(predicted_texts, df_test, metric):
    """
    Evaluates the model predictions using BLEU, chrF, and BERTScore metrics.

    Parameters
    ----------
    df_test : pd.DataFrame
        A DataFrame containing the test data, including reference translations.

    Returns
    -------
    None
        This function does not return any value. It computes and prints the
        metrics for BLEU, chrF, and BERTScore.

    Notes
    -----
    The computed metrics are displayed in the console for review.
    """
    torch.cuda.empty_cache()
    # Convert the list of predicted texts to a NumPy array and flatten it
    # (to ensure proper shape)
    predictions = np.array(predicted_texts).flatten()

    # Extract the reference outputs
    # from the specified DataFrame column using Swifter
    reference = df_test["Arabic"].swifter.apply(lambda text: text[8:])

    # Convert the list of processed reference texts back
    # to a NumPy array for further computation
    reference = np.array(reference).reshape((len(reference), 1))

    # Load the specified evaluation metric
    # from the Hugging Face `evaluate` library
    evaluator = evaluate.load(metric)

    # Compute the metric by comparing predictions to references
    # If the metric is 'bertscore',
    # additional arguments (e.g., language) may be needed
    if metric == "bertscore":
        metric_result = evaluator.compute(predictions=predictions,
                                          references=reference,
                                          lang="en")
    else:
        metric_result = evaluator.compute(predictions=predictions,
                                          references=reference)

    # Return the computed metric result and related statistics
    return metric_result


def mlflow_logging(source_language,
                   target_language,
                   metric_bleu,
                   metric_chrf,
                   metric_bert):

    """
    Logs translation evaluation metrics (BLEU, CHRF, and BERT scores) to an MLflow experiment.

    This function creates or retrieves an MLflow experiment specific to a source-target language pair,
    and logs the provided evaluation metrics. If the experiment already exists, the metrics are appended
    to the latest run; otherwise, a new experiment and run are created.

    Parameters
    ----------
    source_language : str
        The ISO 639-1 code or name of the source language.
    target_language : str
        The ISO 639-1 code or name of the target language.
    metric_bleu : float
        The BLEU score for the translation quality.
    metric_chrf : float
        The CHRF score for the translation quality.
    metric_bert : float
        The BERT score for the semantic similarity.

    Returns
    -------
    None
        This function does not return a value. It logs data directly to MLflow.

    Notes
    -----
    - If an MLflow experiment for the given source-target language pair does not exist, a new experiment is created automatically.
    - Metrics are logged as parameters in the current or newly created MLflow run.
    """


    experiment_name = f"translator_{source_language}_{target_language}_spec"
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    runs = ""

    if experiment:
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        run_id = runs.iloc[0, 0]
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(experiment_name)
        run_id = None

    # Start the MLflow run
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_param("BLEU score", metric_bleu)
        mlflow.log_param("CHRF score", metric_chrf)
        mlflow.log_param("BERT score", metric_bert)


def evaluation(df_test, model, tokenizer, source, target):
    """
    Evaluate the model predictions using BLEU, chrF, and BERTScore metrics.

    Args:
    - df_test: DataFrame containing the test data, including reference translations.

    This function prints the computed metrics for BLEU, chrF, and BERTScore.
    """

    # Generate predicted translations using the model
    predicted_texts = translation(df_test, model, tokenizer, source)

    # Compute the BLEU metric
    metric_bleu = metric_compute(predicted_texts, df_test, "bleu")

    # Compute the chrF metric
    metric_chrf = metric_compute(predicted_texts, df_test, "chrf")

    # Compute the BERTScore metric
    metric_bert = metric_compute(predicted_texts, df_test, "bertscore")

    metric_bert = sum(metric_bert['precision'])/len(metric_bert['precision'])

    mlflow_logging(source,
                   target,
                   metric_bleu,
                   metric_chrf,
                   metric_bert)

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
    # Print first 5 predicted texts as a sample
    for i, prediction in enumerate(predicted_texts[:5]):
        print(f"Prediction {i+1}: {prediction}")
