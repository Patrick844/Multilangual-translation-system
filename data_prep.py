import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from tqdm import tqdm
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments # HF librairies for fine tuning
import pandas as pd # DataFrame manipulation (csv file)
from google.colab import drive
import langcodes
from huggingface_hub import hf_hub_download
import re
import unicodedata
import swifter
import py3langid as langid
import fasttext
tqdm.pandas()

lang_code = {
    'aa': 'aar', 'ab': 'abk', 'af': 'afr', 'ak': 'aka', 'am': 'amh',
    'ar': 'ara', 'an': 'arg', 'as': 'asm', 'av': 'ava', 'ay': 'aym',
    'az': 'aze', 'ba': 'bak', 'be': 'bel', 'bg': 'bul', 'bh': 'bih',
    'bi': 'bis', 'bm': 'bam', 'bn': 'ben', 'bo': 'bod', 'br': 'bre',
    'bs': 'bos', 'ca': 'cat', 'ce': 'che', 'ch': 'cha', 'co': 'cos',
    'cr': 'cre', 'cs': 'ces', 'cu': 'chu', 'cv': 'chv', 'cy': 'cym',
    'da': 'dan', 'de': 'deu', 'dv': 'div', 'dz': 'dzo', 'ee': 'ewe',
    'el': 'ell', 'en': 'eng', 'eo': 'epo', 'es': 'spa', 'et': 'est',
    'eu': 'eus', 'fa': 'fas', 'ff': 'ful', 'fi': 'fin', 'fj': 'fij',
    'fo': 'fao', 'fr': 'fra', 'fy': 'fry', 'ga': 'gle', 'gd': 'gla',
    'gl': 'glg', 'gn': 'grn', 'gu': 'guj', 'gv': 'glv', 'ha': 'hau',
    'he': 'heb', 'hi': 'hin', 'ho': 'hmo', 'hr': 'hrv', 'ht': 'hat',
    'hu': 'hun', 'hy': 'hye', 'hz': 'her', 'ia': 'ina', 'id': 'ind',
    'ie': 'ile', 'ig': 'ibo', 'ii': 'iii', 'ik': 'ipk', 'io': 'ido',
    'is': 'isl', 'it': 'ita', 'iu': 'iku', 'ja': 'jpn', 'jv': 'jav',
    'ka': 'kat', 'kg': 'kon', 'ki': 'kik', 'kj': 'kua', 'kk': 'kaz',
    'kl': 'kal', 'km': 'khm', 'kn': 'kan', 'ko': 'kor', 'kr': 'kau',
    'ks': 'kas', 'ku': 'kur', 'kv': 'kom', 'kw': 'cor', 'ky': 'kir',
    'la': 'lat', 'lb': 'ltz', 'lg': 'lug', 'li': 'lim', 'ln': 'lin',
    'lo': 'lao', 'lt': 'lit', 'lu': 'lub', 'lv': 'lav', 'mg': 'mlg',
    'mh': 'mah', 'mi': 'mri', 'mk': 'mkd', 'ml': 'mal', 'mn': 'mon',
    'mr': 'mar', 'ms': 'msa', 'mt': 'mlt', 'my': 'mya', 'na': 'nau',
    'nb': 'nob', 'nd': 'nde', 'ne': 'nep', 'ng': 'ndo', 'nl': 'nld',
    'nn': 'nno', 'no': 'nor', 'nr': 'nbl', 'nv': 'nav', 'ny': 'nya',
    'oc': 'oci', 'oj': 'oji', 'om': 'orm', 'or': 'ori', 'os': 'oss',
    'pa': 'pan', 'pi': 'pli', 'pl': 'pol', 'ps': 'pus', 'pt': 'por',
    'qu': 'que', 'rm': 'roh', 'rn': 'run', 'ro': 'ron', 'ru': 'rus',
    'rw': 'kin', 'sa': 'san', 'sc': 'srd', 'sd': 'snd', 'se': 'sme',
    'sg': 'sag', 'si': 'sin', 'sk': 'slk', 'sl': 'slv', 'sm': 'smo',
    'sn': 'sna', 'so': 'som', 'sq': 'sqi', 'sr': 'srp', 'ss': 'ssw',
    'st': 'sot', 'su': 'sun', 'sv': 'swe', 'sw': 'swa', 'ta': 'tam',
    'te': 'tel', 'tg': 'tgk', 'th': 'tha', 'ti': 'tir', 'tk': 'tuk',
    'tl': 'tgl', 'tn': 'tsn', 'to': 'ton', 'tr': 'tur', 'ts': 'tso',
    'tt': 'tat', 'tw': 'twi', 'ty': 'tah', 'ug': 'uig', 'uk': 'ukr',
    'ur': 'urd', 'uz': 'uzb', 've': 'ven', 'vi': 'vie', 'vo': 'vol',
    'wa': 'wln', 'wo': 'wol', 'xh': 'xho', 'yi': 'yid', 'yo': 'yor',
    'za': 'zha', 'zh': 'zho', 'zu': 'zul'
}


def get_language_code(language_name: str) -> str:
    """
    Retrieves the ISO 639-3 language code from the given language name using langcodes.

    Args:
        language_name (str): The name of the language (e.g., "French").

    Returns:
        str: The ISO 639-3 code for the language, or None if the language is not found.

    Raises:
        LookupError: If the language code cannot be found.

    Example:
        get_language_code("French")  # Returns 'fra'
    """
    try:
        language = langcodes.find(language_name)
        return language.to_alpha3()  # Returns ISO 639-3 code
    except LookupError:
        return None


def detect_language(row,src,model):
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
        code_input = model.predict(text_input)[0][0].replace("__label__", "").split("_")[0]
        code_target = model.predict(text_target)[0][0].replace("__label__", "").split("_")[0]

        # Return both language codes (English, Arabic)
        return code_input, code_target
    except Exception as e:
        # If detection fails, return None for both
        return None, None

def update_rows_2(row,src,model):
    # Detect the source and target language codes for the row
    code_input, code_target = detect_language(row,src,model)

    if code_target == "arb":
        code_target="ara"

    # Check if both input and target language codes are valid
    if code_input and code_target:
        # Append detected language codes to a list for tracking (optional)

        # Update the 'English' and 'Arabic' columns with the MarianMT language code format
        row[src] = f">>{code_input}<< " + row[src]
        row["Arabic"] = f">>{code_target}<< " + row["Arabic"]


        return row
    else:
        # If language detection fails, return the row unchanged
        return row

# Utility functions used across multiple languages
def remove_extra_whitespace(text):
    return " ".join(text.split())

def lowercase_text(text):
    return text.lower()

### English-specific Preprocessing
def preprocess_english(text):
    text = lowercase_text(text)
    text = handle_english_contractions(text)
    text = remove_extra_whitespace(text)
    return text

def handle_english_contractions(text):
    contractions = {"I'm": "I am", "you're": "you are", "isn't": "is not", "can't": "cannot"}
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text


### French-specific Preprocessing
def preprocess_french(text):
    text = lowercase_text(text)
    text = normalize_french_accents(text)
    text = normalize_french_punctuation(text)
    text = remove_extra_whitespace(text)
    return text

def normalize_french_accents(text):
    return unicodedata.normalize('NFC', text)

def normalize_french_punctuation(text):
    return text.replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!')


### Italian-specific Preprocessing
def preprocess_italian(text):
    text = lowercase_text(text)
    text = normalize_italian_punctuation(text)
    text = remove_extra_whitespace(text)
    return text

def normalize_italian_punctuation(text):
    return text.replace("’", "'")


### Russian-specific Preprocessing
def preprocess_russian(text):
    text = remove_extra_whitespace(text)
    return text


### Turkish-specific Preprocessing
def preprocess_turkish(text):
    text = lowercase_turkish(text)
    text = normalize_turkish_punctuation(text)
    text = remove_extra_whitespace(text)
    return text

def lowercase_turkish(text):
    return text.replace('I', 'ı').replace('İ', 'i').lower()

def normalize_turkish_punctuation(text):
    return text.replace("’", "'")


### Spanish-specific Preprocessing
def preprocess_spanish(text):
    text = lowercase_text(text)
    text = normalize_spanish_accents(text)
    text = remove_extra_whitespace(text)
    return text

def normalize_spanish_accents(text):
    return unicodedata.normalize('NFC', text)


### Greek-specific Preprocessing
def preprocess_greek(text):
    text = lowercase_text(text)
    text = normalize_greek_accents(text)
    text = remove_extra_whitespace(text)
    return text

def preprocess_romania(text):
     # Step 1: Lowercasing
    text = text.lower()
    
    # Step 2: Whitespace and punctuation cleaning
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Removes extra whitespaces
    
    # Step 3: Diacritic normalization (optional, only if you want to remove diacritics)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

def normalize_greek_accents(text):
    return unicodedata.normalize('NFC', text)

def format_table(filepath: str, source: str, output_file: str) -> None:
    """
    Preprocesses a CSV file containing Arabic translations and a source language.
    The function applies normalization, removes diacritics, and processes source and target columns
    for use in MarianMT fine-tuning.

    Args:
        filepath (str): Path to the input CSV file.
        source (str): Name of the source language (e.g., 'French', 'English', 'Russian', etc.).
        output_file (str): Path to the output CSV file where the processed data will be saved.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        ValueError: If the source language column is not present in the CSV.
        KeyError: If the 'Arabic' column is not found in the CSV.
        Exception: For any unexpected errors during processing.

    Example:
        format_table("data_fr.csv", "French", "processed_data.csv")
    """

    try:
        # Step 1: Load the data from the CSV file
        print("Loading data from CSV...")
        df = pd.read_csv(filepath)
        if source not in df.columns or "Arabic" not in df.columns:
            raise KeyError(f"Columns '{source}' and/or 'Arabic' not found in the CSV.")
        print(f"Loaded {len(df)} rows.")

        # Step 2: Drop any rows with missing values
        print("Dropping rows with missing values...")
        df = df.dropna()
        print(f"{len(df)} rows remaining after dropping missing values.")
        print("")  # Print an empty line for clarity

        # Step 3: Apply normalization to the Arabic column
        print("Applying normalization to the Arabic column...")
        df["Arabic"] = df["Arabic"].progress_apply(lambda row: normalize_unicode(row))
        print("Normalization applied.")
        print("")

        # Step 4: Apply diacritic removal to the Arabic column
        print("Removing diacritics from the Arabic column...")
        df["Arabic"] = df["Arabic"].progress_apply(lambda row: dediac_ar(row))
        print("Diacritics removed.")
        print("")

        # Step 5: Remove extra whitespaces from the Arabic column
        print("Removing extra whitespaces from the Arabic column...")
        df["Arabic"] = df["Arabic"].progress_apply(lambda row: remove_extra_whitespace(row))
        print("Extra whitespaces removed from Arabic.")
        print("")

        # Step 6: Preprocess the source language based on its type
        print(f"Applying preprocessing to the source language: {source}...")

        def preprocess_source(row):
            if source.lower() == 'english' or source.lower() == 'eng':
                return preprocess_english(row)
            elif source.lower() == 'french' or source.lower() == 'fra':
                return preprocess_french(row)
            elif source.lower() == 'italian' or source.lower() == 'ita':
                return preprocess_italian(row)
            elif source.lower() == 'russian' or source.lower() == 'rus':
                return preprocess_russian(row)
            elif source.lower() == 'turkish' or source.lower() == 'tur':
                return preprocess_turkish(row)
            elif source.lower() == 'spanish' or source.lower() == 'spa':
                return preprocess_spanish(row)
            elif source.lower() == 'greek' or source.lower() == 'ell':
                return preprocess_greek(row)
            else:
                return row  # Return row unchanged if the language is not supported

        df[source] = df[source].progress_apply(preprocess_source)

        print(f"Preprocessing for {source} applied.")
        print("")

        # Step 7: Remove extra whitespaces from the source column
        print("Removing extra whitespaces from the source column...")
        df[source] = df[source].progress_apply(lambda row: remove_extra_whitespace(row))
        print(f"Extra whitespaces removed from {source}.")
        print("")

        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        fasttext_model = fasttext.load_model(model_path)
        print("FastText model loaded.")

        # Step 8: Update rows for MarianMT fine-tuning by adding language codes
        print("Updating rows for MarianMT fine-tuning...")
        df = df.swifter.apply(lambda row: update_rows_2(row, source,fasttext_model), axis=1)
        print("Rows updated for MarianMT fine-tuning.")
        print("")

        # Step 9: Get the language code for the source language
        code = get_language_code(source)

        # Step 10: Apply mask for filtering rows where certain conditions are met
        print("Applying mask for specific Arabic and source string slices...")
        mask = (df["Arabic"].str.contains(r">>ara<<")) & (df[source].str.contains(f">>{code}<<"))
        df = df[mask]
        print(f"{len(df)} rows remaining after applying the mask.")
        print("")

        # Step 11: Save the processed DataFrame to a new CSV file
        print("Saving the processed data to a CSV file...")
        df.to_csv(output_file, index=False)
        print(f"Data successfully saved to '{output_file}'.")
        print("Data processing complete.")

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    except KeyError as e:
        raise KeyError(f"Missing necessary column: {e}")
    except Exception as e:
            print(f"An error occurred: {e}")
            raise


