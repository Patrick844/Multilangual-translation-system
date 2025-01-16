
"""
Module Name: data_prep.py

This module provides functionality for preprocessing text data in various languages,
formatting dataframes, and preparing datasets for training machine translation models
like MarianMT.

Features
--------
- Language-specific preprocessing functions for:
  - English
  - French
  - Italian
  - Russian
  - Turkish
  - Spanish
  - Greek
  - Romanian
- General utility functions for:
  - Removing whitespace
  - Expanding abbreviations
  - Normalizing text (e.g., punctuation, accents)
- Automatic language detection using FastText.
- Integration with MLflow for tracking data inputs.
- Processes source and target language columns for fine-tuning MarianMT.

Main Classes and Functions
--------------------------
Text Preprocessing Functions:
    - ``preprocess_english(text)``: Handles English-specific preprocessing.
    - ``preprocess_french(text)``: Handles French-specific preprocessing.
    - ``preprocess_italian(text)``: Handles Italian-specific preprocessing.
    - ``preprocess_russian(text)``: Handles Russian-specific preprocessing.
    - ``preprocess_turkish(text)``: Handles Turkish-specific preprocessing.
    - ``preprocess_spanish(text)``: Handles Spanish-specific preprocessing.
    - ``preprocess_greek(text)``: Handles Greek-specific preprocessing.
    - ``preprocess_romania(text)``: Handles Romanian-specific preprocessing.

Utility Functions:
    - ``get_language_code(language_name)``: Retrieves ISO 639-3 language codes.
    - ``detect_language(row, src, model)``: Detects source and target languages in a DataFrame row.
    - ``rm_whitespace(text)``: Removes unnecessary whitespace from text.
    - ``handle_english_contractions(text)``: Expands English contractions.
    - ``expand_abbreviations(text)``: Expands abbreviations using a dictionary.
    - ``format_table(filepath, source, output_file)``: Processes a CSV file, normalizes text, and prepares data for training.

Dependencies
------------
- ``pandas``: For handling tabular data.
- ``tqdm``: For progress bars during processing.
- ``camel_tools``: For Arabic text normalization and diacritic removal.
- ``langcodes``: For handling language code lookups.
- ``huggingface_hub``: For downloading FastText models.
- ``fasttext``: For language detection.
- ``mlflow``: For tracking input data for machine learning pipelines.

Example Usage
-------------
.. code-block:: python

    from data_prep import format_table

    # Preprocess a CSV file for fine-tuning MarianMT
    format_table(
        filepath="input_data.csv",
        source="English",
        output_file="processed_data.csv"
    )
"""

import re
import unicodedata
import pandas as pd
from tqdm import tqdm
import swifter
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode as n_unicode
import langcodes
from huggingface_hub import hf_hub_download
import fasttext
import mlflow
from mlflow.data import from_pandas
from mlflow.data.pandas_dataset import PandasDataset
from lingowiz.utils import abbreviation_dict

tqdm.pandas()


def get_language_code(language_name: str) -> str:
    """Retrieves the ISO 639-3 language code from the given language name using
    langcodes.

    Parameters
    ----------
    language_name : str
        The name of the language (e.g., "French").

    Returns
    -------
    str
        The ISO 639-3 code for the language, or ``None`` if the language is not found.

    Raises
    ------
    LookupError
        If the language code cannot be found.

    Examples
    --------
    .. code-block:: python

        get_language_code("French")  # Returns 'fra'
    """


    try:
        language = langcodes.find(language_name)
        return language.to_alpha3()  # Returns ISO 639-3 code
    except LookupError:
        return None


def detect_language(row, src, model):

    """Performs automatic language detection and filters out invalid
    detections.

    This function detects the source and target languages in a given row using a language
    detection model. If an error occurs during detection, the function returns ``(None, None)``.

    Parameters
    ----------
    row : dict
        A single row containing source and target language text.
    src : str
        The name of the source column in the row.
    model : object
        The model used for language detection, expected to have a ``predict`` method.

    Returns
    -------
    tuple
        A tuple containing:
        - str: Code of the source language.
        - str: Code of the target language.

    Raises
    ------
    KeyError
        If the required keys are missing from the row dictionary.
    AttributeError
        If the model object does not have a ``predict`` method.
    ValueError
        If the text preprocessing or model output is invalid.
    Exception
        For any other unexpected errors.

    Notes
    -----
    In case of an error during language detection, the function returns ``(None, None)``
    instead of raising the exception.
    """


    try:

        # Preprocess the text for English column
        text_input = re.sub(r'\b[A-Z][a-z]*\b', '', row[src])
        text_input = unicodedata.normalize('NFC', text_input)
        text_input = text_input.lower()

        # Preprocess the text for Arabic column
        text_target = re.sub(r'\b[A-Z][a-z]*\b', '', row["Arabic"])
        text_target = unicodedata.normalize('NFC', text_target)
        text_target = text_target.lower()

        # Detect the language for both columns
        code_input = model.predict(text_input)[0][0]
        code_input = code_input.replace("__label__", "").split("_")[0]

        code_target = model.predict(text_target)[0][0]
        code_target = code_target.replace("__label__", "").split("_")[0]

        # Return both language codes (English, Arabic)
        return code_input, code_target

    except KeyError as e:
        # Handle missing keys in the row
        print(f"KeyError: {e}")
        return None, None
    except AttributeError as e:
        # Handle issues with `model` or its methods
        print(f"AttributeError: {e}")
        return None, None
    except ValueError as e:
        # Handle unexpected values in the text or prediction
        print(f"ValueError: {e}")
        return None, None
    except Exception as e:
        # Log unexpected exceptions (optional)
        print(f"Unexpected error: {e}")
        return None, None


def update_rows_2(row, src, model):

    """
    Updates a DataFrame row with language-specific tags for MarianMT fine-tuning.

    This function detects the source and target languages of a given row using a 
    language detection model. It appends the detected language codes as MarianMT 
    language tags (e.g., ``>>eng<<``) to the source and target text columns. If the 
    target language is detected as ``arb``, it is normalized to ``ara``.

    Parameters
    ----------
    row : pandas.Series or dict
        A single row of the DataFrame containing source and target language text.
    src : str
        The name of the source language column in the row (e.g., ``English``).
    model : object
        The language detection model, expected to have a ``predict`` method.

    Returns
    -------
    pandas.Series or dict
        The updated row with language tags appended to the source and target text.

    Notes
    -----
    - The target language code ``arb`` is replaced with ``ara`` for consistency.
    - If language detection fails (i.e., ``code_input`` or ``code_target`` is ``None``), the function returns the original row without modification.

    Examples
    --------
    .. code-block:: python

        row = {"English": "Hello", "Arabic": "مرحبا"}
        updated_row = update_rows_2(row, src="English", model=language_model)
        print(updated_row)
    """

    # Detect the source and target language codes for the row
    code_input, code_target = detect_language(row, src, model)

    if code_target == "arb":
        code_target = "ara"

    # Check if both input and target language codes are valid
    if code_input and code_target:
        # Append detected language codes to a list for tracking (optional)

        # Update the 'English' and 'Arabic' columns
        # with the MarianMT language code format
        row[src] = f">>{code_input}<< " + row[src]
        row["Arabic"] = f">>{code_target}<< " + row["Arabic"]

        return row

    return row


def rm_whitespace(text) -> str:
    """Removes unnecessary whitespace from the input text.

    Parameters
    ----------
    text : str
        The input text from which to remove unnecessary whitespace.

    Returns
    -------
    str
        The text with unnecessary whitespace removed.
    """


    # Utility functions used across multiple languages
    return " ".join(text.split())


def lowercase_text(text):
    """Converts the input text to lowercase.

    Parameters
    ----------
    text : str
        The input text to convert to lowercase.

    Returns
    -------
    str
        The text converted to lowercase.
    """
    return text.lower()


def preprocess_english(text):
    """Preprocesses English text by applying multiple transformations.

    Parameters
    ----------
    text : str
        The input text in English.

    Returns
    -------
    str
        The preprocessed text after applying all English-specific transformations.
    """
    # English-specific Preprocessing
    text = process_medical_data(text)
    text = handle_english_contractions(text)
    text = rm_whitespace(text)

    return text


def handle_english_contractions(text):
    """Expands English contractions in the text.

    Parameters
    ----------
    text : str
        The input text in English.

    Returns
    -------
    str
        The text with contractions expanded.
    """
    # Normalizing English Contractions
    contractions = {
        "I'm": "I am",
        "you're": "you are",
        "isn't": "is not",
        "can't": "cannot"
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text


def preprocess_french(text):
    """Preprocesses French text by applying specific transformations.

    Parameters
    ----------
    text : str
        The input text in French.

    Returns
    -------
    str
        The preprocessed text after applying all French-specific transformations.
    """
    text = lowercase_text(text)
    text = normalize_french_accents(text)
    text = normalize_french_punctuation(text)
    text = rm_whitespace(text)
    return text


def normalize_french_accents(text):
    """Standardizes French accents in the text.

    Parameters
    ----------
    text : str
        The input text in French.

    Returns
    -------
    str
        The text with accents normalized.
    """
    return unicodedata.normalize('NFC', text)


def normalize_french_punctuation(text):
    """Normalizes French punctuation in the text.

    Parameters
    ----------
    text : str
        The input text in French.

    Returns
    -------
    str
        The text with punctuation normalized.
    """
    text = text.replace(' :', ':')
    text = text.replace(' ;', ';')
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    return text


def preprocess_italian(text):
    """Preprocesses Italian text by applying specific transformations.

    Parameters
    ----------
    text : str
        The input text in Italian.

    Returns
    -------
    str
        The preprocessed text after applying all Italian-specific transformations.
    """
    text = lowercase_text(text)
    text = normalize_italian_punctuation(text)
    text = rm_whitespace(text)
    return text


def normalize_italian_punctuation(text):
    """Normalizes Italian punctuation in the text.

    Parameters
    ----------
    text : str
        The input text in Italian.

    Returns
    -------
    str
        The text with normalized Italian punctuation.
    """
    return text.replace("’", "'")


def preprocess_russian(text):
    """Preprocesses Russian text by removing unnecessary whitespace.

    Parameters
    ----------
    text : str
        The input text in Russian.

    Returns
    -------
    str
        The preprocessed text with whitespace removed.
    """
    text = rm_whitespace(text)
    return text


def preprocess_turkish(text):
    """Preprocesses Turkish text by applying specific transformations.

    Parameters
    ----------
    text : str
        The input text in Turkish.

    Returns
    -------
    str
        The preprocessed text after applying all Turkish-specific transformations.
    """
    text = lowercase_turkish(text)
    text = normalize_turkish_punctuation(text)
    text = rm_whitespace(text)
    return text


def lowercase_turkish(text):
    """Converts Turkish text to lowercase, handling special Turkish characters.

    Parameters
    ----------
    text : str
        The input text in Turkish.

    Returns
    -------
    str
        The text converted to lowercase with Turkish-specific characters handled.
    """
    return text.replace('I', 'ı').replace('İ', 'i').lower()


def normalize_turkish_punctuation(text):
    """Normalizes Turkish punctuation in the text.

    Parameters
    ----------
    text : str
        The input text in Turkish.

    Returns
    -------
    str
        The text with normalized Turkish punctuation.
    """
    return text.replace("’", "'")


def preprocess_spanish(text):
    """Preprocesses Spanish text by applying specific transformations.

    Parameters
    ----------
    text : str
        The input text in Spanish.

    Returns
    -------
    str
        The preprocessed text after applying all Spanish-specific transformations.
    """
    text = lowercase_text(text)
    text = normalize_spanish_accents(text)
    text = rm_whitespace(text)
    return text


def normalize_spanish_accents(text):
    """Normalizes Spanish accents in the text.

    Parameters
    ----------
    text : str
        The input text in Spanish.

    Returns
    -------
    str
        The text with normalized Spanish accents.
    """
    return unicodedata.normalize('NFC', text)


def preprocess_greek(text):
    """Preprocesses Greek text by applying specific transformations.

    Parameters
    ----------
    text : str
        The input text in Greek.

    Returns
    -------
    str
        The preprocessed text after applying all Greek-specific transformations.
    """
    text = lowercase_text(text)
    text = normalize_greek_accents(text)
    text = rm_whitespace(text)
    return text


def preprocess_romania(text):
    """Preprocesses Romanian text by applying transformations such as
    lowercasing, punctuation removal, whitespace cleaning, and optional
    diacritic normalization.

    Parameters
    ----------
    text : str
        The input text in Romanian.

    Returns
    -------
    str
        The preprocessed, normalized, and cleaned text.
    """
    # Step 1: Lowercasing
    text = text.lower()

    # Step 2: Whitespace and punctuation cleaning
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Removes extra whitespaces

    # Step 3: Diacritic normalization (optional)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text


def normalize_greek_accents(text):
    """Normalizes Greek text accents using Unicode NFC normalization.

    Parameters
    ----------
    text : str
        The input text in Greek.

    Returns
    -------
    str
        The text with normalized Greek accents.
    """
    return unicodedata.normalize('NFC', text)



def process_medical_data(data):
    """Processes medical data by normalizing symbols, converting to lowercase,
    expanding abbreviations, and adding spaces between letters and numbers.

    Parameters
    ----------
    data : str
        The input medical data text.

    Returns
    -------
    str
        The processed and normalized medical data.

    Notes
    -----
    This function ensures that medical data is in a standardized format
    for downstream processing.
    """
    data = data.replace(".", " ")
    data = data.replace("=", " ")
    data = data.replace("-", " ")
    data = data.replace("_", " ")
    data = data.lower()
    data = expand_abbreviations(data)
    data = add_space_between_letters_and_numbers(data)
    return data


def add_space_between_letters_and_numbers(text):
    """Adds spaces between letters and numbers in the text.

    Parameters
    ----------
    text : str
        The input text containing letters and numbers.

    Returns
    -------
    str
        The text with spaces inserted between letters and numbers.

    Notes
    -----
    Uses regular expressions to identify and separate letters from numbers.
    """
    separated_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    return separated_text


def expand_abbreviations(text):
    """Expands abbreviations in the text based on a predefined abbreviation
    dictionary.

    Parameters
    ----------
    text : str
        The input text containing abbreviations.

    Returns
    -------
    str
        The text with abbreviations expanded.

    Notes
    -----
    This function uses a dictionary (`abbreviation_dict`) to replace
    abbreviations with their corresponding expanded forms.

    Raises
    ------
    KeyError
        If the abbreviation key is not found in the dictionary.
    """
    pattern = re.compile(r'\b(' +
                         '|'.join(re.escape(key) for key in abbreviation_dict.keys()) +
                         r')\b')
    expanded_text = pattern.sub(lambda x: abbreviation_dict[x.group()], text)
    return expanded_text


def format_table(filepath: str, source: str, output_file: str) -> None:
    """Preprocesses a CSV file for MarianMT fine-tuning by normalizing text and
    adding language-specific tags.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file.
    source : str
        Name of the source language (e.g., 'French').
    output_file : str
        Path to save the processed CSV file.

    Returns
    -------
    None
        The function writes the processed data directly to the output file.

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.
    KeyError
        If the required 'Arabic' column or the source language column is missing.
    Exception
        For any unexpected errors during processing.

    Examples
    --------
    .. code-block:: python

        format_table("data_fr.csv", "French", "processed_data.csv")
    """

    try:
        # Step 1: Load the data from the CSV file
        print("Loading data from CSV...")
        df = pd.read_csv(filepath)
        if source not in df.columns or "Arabic" not in df.columns:
            error = f"Columns '{source}' and/or 'Arabic' not found in the CSV."
            raise KeyError(error)
        print(f"Loaded {len(df)} rows.")

        # Step 2: Drop any rows with missing values
        print("Dropping rows with missing values...")
        df = df.dropna()
        print(f"{len(df)} rows remafining after dropping missing values.")
        print("")  # Print an empty line for clarity

        # Step 3: Apply normalization to the Arabic column
        print("Applying normalization to the Arabic column...")
        df_arabic = df["Arabic"]
        df_arabic = df_arabic.progress_apply(n_unicode)
        print("Normalization applied.")
        print("")

        # Step 4: Apply diacritic removal to the Arabic column
        print("Removing diacritics from the Arabic column...")
        df_arabic = df_arabic.progress_apply(dediac_ar)
        print("Diacritics removed.")
        print("")

        # Step 5: Remove extra whitespaces from the Arabic column
        print("Removing extra whitespaces from the Arabic column...")
        df_arabic = df_arabic.progress_apply(rm_whitespace)
        print("Extra whitespaces removed from Arabic.")
        print("")

        df["Arabic"] = df_arabic

        # Step 6: Preprocess the source language based on its type
        print(f"Applying preprocessing to the source language: {source}...")

        def preprocess_source(row):
            if source.lower() == 'english' or source.lower() == 'eng':
                return preprocess_english(row)
            if source.lower() == 'french' or source.lower() == 'fra':
                return preprocess_french(row)
            if source.lower() == 'italian' or source.lower() == 'ita':
                return preprocess_italian(row)
            if source.lower() == 'russian' or source.lower() == 'rus':
                return preprocess_russian(row)
            if source.lower() == 'turkish' or source.lower() == 'tur':
                return preprocess_turkish(row)
            if source.lower() == 'spanish' or source.lower() == 'spa':
                return preprocess_spanish(row)
            if source.lower() == 'greek' or source.lower() == 'ell':
                return preprocess_greek(row)
            return preprocess_english(row)

        df[source] = df[source].progress_apply(preprocess_source)

        print(f"Preprocessing for {source} applied.")
        print("")

        # Step 7: Remove extra whitespaces from the source column
        print("Removing extra whitespaces from the source column...")
        df[source] = df[source].progress_apply(rm_whitespace)
        print(f"Extra whitespaces removed from {source}.")
        print("")
        repo_id = "facebook/fasttext-language-identification"
        model_path = hf_hub_download(repo_id=repo_id,
                                     filename="model.bin")
        fasttext_model = fasttext.load_model(model_path)
        print("FastText model loaded.")

        # Step 8: Update rows for MarianMT fine-tuning by adding language codes
        # print("Updating rows for MarianMT fine-tuning...")
        # df = df.swifter.apply(lambda row: update_rows_2(row,
        #                                                 source,
        #                                                 fasttext_model),
        #                       axis=1)
        # print("Rows updated for MarianMT fine-tuning.")
        # print("")

        # # Step 9: Get the language code for the source language
        # code = get_language_code(source)

        # # Step 10: Apply mask for filtering rows
        # print("Applying mask for specific Arabic and source string slices...")
        # mask_arabic = df["Arabic"].str.contains(r">>ara<<")
        # mask_source = df[source].str.contains(f">>{code}<<")
        # df = df[mask_arabic & mask_source]
        # print(f"{len(df)} rows remaining after applying the mask.")
        # print("")

        # Step 11: Save the processed DataFrame to a new CSV file
        print("Saving the processed data to a CSV file...")
        df.to_csv(output_file, index=False)
        dataset: PandasDataset = from_pandas(df,
                                             source=output_file)
        mlflow.log_input(dataset, context="training")
        print(f"Data successfully saved to '{output_file}'.")
        print("Data processing complete.")

    except FileNotFoundError as exc:
        # Re-raise the exception with additional context, preserving the original traceback
        raise FileNotFoundError(f"The file {filepath} does not exist.") from exc
    except KeyError as exc:
        # Re-raise the KeyError with context about the missing column
        raise KeyError(f"Missing necessary column: {exc}") from exc
    except Exception as exc:
        # Log and re-raise the unexpected exception
        print(f"An error occurred: {exc}")
        raise exc
