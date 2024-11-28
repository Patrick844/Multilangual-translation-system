

"""
Module Name: inference_pipeline.py

This module provides a complete pipeline for processing, translating, and evaluating
text in multiple languages using fine-tuned MarianMT models. It includes utilities for
language detection, preprocessing, translation, confidence computation, grammar error
analysis, and logging results to MLflow.

Features:
---------
- **Language Detection**: Automatically detect the language of the input text using FastText.
- **Preprocessing**: Clean and preprocess text for specific languages before translation.
- **Translation**: Perform direct or indirect translation using MarianMT models and tokenizers.
- **Confidence Calculation**: Compute confidence scores for generated translations.
- **Grammar Checking**: Evaluate translations using grammar error analysis with LanguageTool.
- **MLflow Logging**: Log translation results, ratings, and metadata to MLflow experiments.
- **Chunking**: Handle large text by splitting it into manageable chunks for translation.

Dependencies:
-------------
- ``transformers``: For MarianMT models and tokenizers.
- ``fasttext``: For language detection.
- ``language_tool_python``: For grammar error checking.
- ``mlflow``: For logging translations and metadata.
- ``huggingface_hub``: For downloading models.
- ``torch``: For confidence computation.
- ``tqdm``: For progress bars during processing.

Usage Example:
--------------
.. code-block:: python

    from inference_pipeline import inference

    text = "Bonjour tout le monde!"
    translation, source_languages, ratings = inference(text)

    print(f"Translation: {translation}")
    print(f"Source Languages: {source_languages}")
    print(f"Ratings: {ratings}")
"""

import re
import warnings
import os
import pandas as pd
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from huggingface_hub import hf_hub_download
import fasttext
from dotenv import load_dotenv
import langcodes
import language_tool_python
import mlflow
import torch
from lingowiz.utils import model_dict
from lingowiz.inference_preprocessing import TextPreprocessor
indirect_translation_language = ["ron"]

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load languages tool
tools = [language_tool_python.LanguageTool('en'),
         language_tool_python.LanguageTool('ar')]

# Environment Variables
token = os.getenv('HF_TOKEN')

# Initialize
tqdm.pandas()
load_dotenv()

print("Loading the FastText language detection model...")


def detect_language(text):
    """
    Detects the language of the input text using a FastText model.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: ISO 639-1 language code for the detected language.
             Defaults to 'eng' if the language is not recognized.
    """
    print("FastText model loaded.")
    repo_id = "facebook/fasttext-language-identification"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.bin")
    fasttext_model = fasttext.load_model(model_path)
    # Detect the language for both columns
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    code = fasttext_model.predict(text)[0][0]
    code = code.replace("__label__", "")
    code = code.split("_")[0]

    # If code not found, default is english
    if code in list(model_dict.keys()):
        pass
    else:
        code = "eng"

    # Return both language codes (English, Arabic)
    return code


def processing(text, code):

    """
      Preprocesses the text based on the detected language and retrieves
      the appropriate model configuration.

      Args:
          text (str): The input text to preprocess.
          code (str): The ISO 639-1 code of the detected language.

      Returns:
          tuple: A tuple containing:
              - str: Preprocessed text.
              - dict: Model configuration for translation.
      """
    processeur = TextPreprocessor(code)
    text = processeur.process(text)

    print("Finding Model ...")
    model_config = model_dict[code]
    return text, model_config


def compute_confidence(scores):
    """
      Computes the confidence score for a translation
      based on model output scores.

      Args:
          scores (list): List of token logits from the model output.

      Returns:
          float: Average confidence score (0.0 to 1.0).
    """
    total_confidence = 0
    for step_logits in scores:
        # Convert logits to probabilities
        probabilities = torch.softmax(step_logits, dim=-1)
        # Get the max probability for each token
        max_prob, _ = torch.max(probabilities, dim=-1)
        # Average max probability per step
        total_confidence += torch.mean(max_prob).item()

    # Normalize confidence score
    avg_confidence = total_confidence / len(scores)

    print(f"Confidence Score: {avg_confidence:.2f}")
    return avg_confidence


def compute_rating(tool, translated_text, total_confidence):
    """
    Computes a rating for the translated text based on
    grammar errors and model confidence.

    Args:
        tool (LanguageTool): The grammar checker tool.
        translated_text (str): The text to evaluate.
        total_confidence (float): Confidence score from the translation model.

    Returns:
        float: A rating between 0 and 10 for the translation quality.
    """
    print("Checking Error ...")

    matches = tool.check(translated_text)
    number_errors = len(matches)

    length_sentence = len(translated_text.split())
    if length_sentence == 0:
        print("The translated text is empty!")
        return 0  # Return a default rating if text is empty

    # Calculate error rate
    grammar_error_rate = number_errors / length_sentence

    # Lack of confidence is penalized (1 - confidence)
    no_confidence = 1 - total_confidence

    # Combine both penalties: grammar error rate and lack of confidence
    rating = 1 - (grammar_error_rate + no_confidence)

    # Ensure rating is between 0 and 1
    rating = max(0, min(1, rating))

    # Optional: Scale the rating to 0–10 for readability
    rating_scaled = rating * 10  # For a scale of 0 to 10

    print(f"\nNumber of Errors: {number_errors} "
          f"\nSentence Length: {length_sentence} "
          f"\nRating: {rating_scaled}")
    return rating_scaled


def mlflow_logging(source_language,
                   target_language,
                   original,
                   translation,
                   rating):

    """
      Logs translation results and metadata to an MLflow experiment.

      Args:
          source_language (str): The language of the original text.
          target_language (str): The language of the translation.
          original (str): The original input text.
          translation (str): The translated text.
          rating (float): Quality rating for the translation.
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

    # New data to be added
    new_data = [{"Original": original,
                 "Translation": translation,
                 "Source": source_language,
                 'Rating': rating}]

    # Start the MLflow run
    with mlflow.start_run(run_id=run_id, nested=True):

        # List artifacts for the given run_id
        artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)

        # Check if 'inference_data' is already logged
        artifact_names = [artifact.path for artifact in artifacts]
        # Adjust this path if the artifact was stored in a subdirectory
        artifact_path = "inference_data.csv"

        if artifact_path in artifact_names:

            # Load Existing Atrifact
            params = {"artifact_path": artifact_path,
                      "run_id": run_id}

            local_artifact_path = mlflow.artifacts.download_artifacts(**params)
            df = pd.read_csv(local_artifact_path)

            # Add new data to existing inference data
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df])

            # Log new inference data on mlflow
            print("Addindg data to inference_data.csv")

        else:
            # Define the data for the DataFrame
            df = pd.DataFrame(new_data)

            # Log the CSV file as an artifact
            print("Creating new DF and logging as artifact")

        # Save the DataFrame to a CSV file
        df.to_csv("inference_data.csv", index=False)
        mlflow.log_artifact("inference_data.csv")


def generating_translation(tokenizer,
                           model,
                           data,
                           num_beams=5,
                           max_length=512):

    """
    Generates translations for the input text using a MarianMT model.

    Args:
        tokenizer (MarianTokenizer): The tokenizer for the MarianMT model.
        model (MarianMTModel): The MarianMT model for translation.
        data (str): The input text to translate.
        num_beams (int, optional): Number of beams. Defaults to 5.
        max_length (int, optional): Max sequence length. Defaults to 512.

    Returns:
        tuple: A tuple containing:
            - str: The translated text.
            - float: Confidence score for the translation.
    """

    # Tokenize data (tansform text into number,
    # same transformation as training)
    inputs = tokenizer(data,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=max_length)

    # Perform translation (inference)
    translated_tokens = model.generate(**inputs,
                                       num_beams=num_beams,
                                       max_length=512,
                                       no_repeat_ngram_size=3,
                                       return_dict_in_generate=True,
                                       output_scores=True)
    scores = translated_tokens.scores
    confidence = compute_confidence(scores)
    translated_tokens = translated_tokens.sequences

    # Decode the generated tokens to human-readable text
    translated_text = tokenizer.decode(translated_tokens[0],
                                       skip_special_tokens=True)

    return translated_text, confidence


def initializing_model(model_name):
    """
      Initializes and loads a MarianMT model.

      Args:
          model_name (str): The name or path of the MarianMT model.

      Returns:
          MarianMTModel: The loaded MarianMT model.
    """
    model = MarianMTModel.from_pretrained(model_name, token=token)
    return model


def initializing_tokenizer(tokenizer_name):
    """
    Initializes and loads a MarianMT tokenizer.

    Args:
        tokenizer_name (str): The name or path of the MarianMT tokenizer.

    Returns:
        MarianTokenizer: The loaded tokenizer.
    """
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name, token=token)
    return tokenizer


def indirect_translation(data,
                         model_config,
                         translation_list,
                         confidence_list,
                         max_length=512,
                         num_beams=5):
    """
    Performs indirect translation via an intermediate language
    (e.g., Romanian to English to Arabic).

    Args:
        data (str): The input text for translation.
        model_config (dict): Model and Tokenizer config.
        translation_list (list): List to store translations.
        confidence_list (list): List to store confidence scores.
        max_length (int, optional): Max sequence length. Defaults to 512.
        num_beams (int, optional): Number of beams. Defaults to 5.

    Returns:
        tuple:
          - Updated translation list
          - Updated confidence list.
    """
    print("")
    print("Indirect ...")
    print("")
    translation_list, confidence_en = direct_translation(data,
                                                         model_config,
                                                         translation_list,
                                                         confidence_list)

    # Initialize additional model and tokenizer (English - Arabic)
    tokenizer_ar = initializing_tokenizer(model_config[2])
    model_ar = initializing_model(model_config[3])

    translated_text, confidence_ar = generating_translation(tokenizer_ar,
                                                            model_ar,
                                                            translation_list[0])
    confidence_list.append(confidence_ar)

    if len(translation_list) == 1:
        translation_list.append(translated_text)
    else:
        translation_list[1] = translation_list[1] + translated_text

    return translation_list, confidence_list


def direct_translation(data,
                       model_config,
                       translation_list,
                       confidence_list,
                       max_length=512,
                       num_beams=5):
    """
    Function to load a fine-tuned MarianMT model
    and perform translation on a DataFrame column with a progress bar.

    Args:
        data (DataFrame): DataFrame containing the texts to be translated.
        model_name (list): List of Model and Tokenizer for specific language
        source_column (str): The name of the column in the DataFrame containing
            the text in the source language.
        max_length (int): Maximum length of input sequences. Defaults to 128.
        num_beams (int): Number of beams for beam search. Defaults to 3.

    Returns:
        tuple:
          - Updated translation list
          - Updated confidence list.
    """

    # Load the tokenizer and the fine-tuned model
    tokenizer = initializing_tokenizer(model_config[1])
    model = initializing_model(model_config[0])

    if len(model_config) > 2:
        # Format data for model
        data = ">>eng<< " + data

    print("Translating ...")
    translated_text, confidence = generating_translation(tokenizer,
                                                         model,
                                                         data)
    confidence_list.append(confidence)

    if not translation_list:
        translation_list.append(translated_text)
    else:
        translation_list[0] = translation_list[0] + translated_text
    return translation_list, confidence_list


def chunk_data(chunk_size, text):
    """
    Splits the input text into smaller chunks for easier processing.

    Args:
        chunk_size (int): Maximum number of words per chunk.
        text (str): The input text to split.

    Returns:
        list: List of text chunks.
    """
    chunk_text = ""
    chunk_list = []
    for i in range(0, len(text.split(" ")), chunk_size):
        if i + chunk_size > len(text.split(" ")):
            chunk_text = text.split(" ")[i:]
        else:
            chunk_text = text.split(" ")[i:i + chunk_size]
        chunk_list.append(" ".join(chunk_text))
    return chunk_list


def process_chunk(chunk_list,
                  code,
                  translation_func,
                  translation_list,
                  confidence_list):
    """
      Processes text chunks for translation.

      Args:
          chunk_list (list): List of text chunks.
          code (str): Language code for the input text.
          translation_func (function): The translation function
            (direct or indirect).
          translation_list (list): List to store translations.
          confidence_list (list): List to store confidence scores.

      Returns:
          tuple:
            - Updated translation list
            - Updated confidence list.
    """
    for chunk in chunk_list:
        print("Processing Text...")
        # Processing
        processed_text, model_config = processing(chunk, code)
        # Translating
        translation_list, confidence_list = translation_func(processed_text,
                                                             model_config,
                                                             translation_list,
                                                             confidence_list)
        print(translation_list)
        print("")

    return translation_list, confidence_list


def inference(text):

    """
      Performs full inference for translation, including language detection,
      preprocessing, translation, and rating computation.

      Args:
          text (str): The input text to translate.

      Returns:
          tuple: A tuple containing:
              - list: Translations in the target language(s).
              - list: Detected source languages.
              - list: Ratings for translation quality.
    """

    # Initializing lists
    source_list = []
    translation_list = []
    confidence_list = []
    confidence_en = []
    confidence_ar = []
    rating_list = []

    # Detecting Language
    print("Detecting Language ...")
    code = detect_language(text)
    source_language = langcodes.get(code).language_name()

    chunk_size = 100
    if code in indirect_translation_language:
        print("")
        print(f"Indirect Translation {source_language} to English to Arabic")
        print("")
        if not source_list:
            source_list.append(source_language)
            source_list.append("English")

        # Indirect Translation, update source language
        # Translation 1 - Source - English
        source_language = source_list[0]

        if len(text.split(" ")) > chunk_size:
            # Chunking data if text too big
            chunk_list = chunk_data(chunk_size, text)
            # Process chunk data
            translation_list, confidence_list = process_chunk(chunk_list,
                                                              code,
                                                              indirect_translation,
                                                              translation_list,
                                                              confidence_list)
            for i in range(len(confidence_list)):
                if i % 2 == 0:
                    confidence_ar.append(confidence_list[i])
                else:
                    confidence_en.append(confidence_list[i])
            confidence_list = []
            confidence_list.append(sum(confidence_ar)/len(confidence_ar))
            confidence_list.append(sum(confidence_en)/len(confidence_en))
        else:
            print("Processing Text ...")
            processed_text, model_config = processing(text, code)  # Processing
            translation_list, confidence_list = indirect_translation(processed_text,
                                                                     model_config,
                                                                     translation_list,
                                                                     confidence_list)

        # Translation 2 - English - Arabic
        source_language = source_list[1]

    else:
        print(f"Direct Translation {source_language} to Arabic")
        print("")
        if not source_list:
            source_list.append(source_language)

        # Direct translation
        # Translation Source - Arabic
        source_language = source_list[0]

        if len(text.split(" ")) > chunk_size:
            chunk_list = chunk_data(chunk_size, text)  # Chunking data if text too big
            translation_list, confidence_list = process_chunk(chunk_list,
                                                              code,
                                                              direct_translation,
                                                              translation_list,
                                                              confidence_list)  # process chunk data
            confidence = sum(confidence_list) / len(confidence_list)
            confidence_list = []
            confidence_list.append(confidence)

        else:
            print("Processing Text ...")
            processed_text, model_config = processing(text, code)
            translation_list, confidence_list = direct_translation(processed_text,
                                                                   model_config,
                                                                   translation_list,
                                                                   confidence_list)

    if len(confidence_list) == 2:
        for i in range(2):
            translated_text = translation_list[i]
            confidence = confidence_list[i]
            tool = tools[i]
            rating = compute_rating(tool,
                                    translated_text,
                                    confidence)
            rating_list.append(rating)
            source_language = source_list[i]
            if i == 1:
                text = translation_list[0]
                target = "Arabic"
            else:
                target = "English"
            print("Logging ...")
            mlflow_logging(source_language,
                           target,
                           text,
                           translated_text,
                           rating)

    else:
        translated_text = translation_list[0]
        confidence = confidence_list[0]
        tool = tools[0]
        source_language = source_list[0]
        rating = compute_rating(tool,
                                translated_text,
                                confidence)
        rating_list.append(rating)

        print("Logging ...")
        mlflow_logging(source_language,
                       "Arabic",
                       text,
                       translated_text,
                       rating)
        print("Logging Complete")

    return translation_list, source_list, rating_list
