# Importing Librairies
import gzip  # Used for file decompression
import shutil # Used for file decompression
from lxml import etree #Used for xml parsing
import csv # Used to convert the file to CSV
import string
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from tqdm import tqdm
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from farasa.segmenter import FarasaSegmenter
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments # HF librairies for fine tuning
import pandas as pd # DataFrame manipulation (csv file)
import langcodes
from huggingface_hub import hf_hub_download
import py3langid as langid
import re
import unicodedata
import os
import fasttext
from dotenv import load_dotenv


load_dotenv()

os.environ['CURL_CA_BUNDLE'] = ""
tqdm.pandas()

print("Loading the FastText language detection model...")
print("FastText model loaded.")
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
fasttext_model = fasttext.load_model(model_path)


class TextPreprocessor:
    def __init__(self, language):
        self.language = language

    def process(self, text):
        if self.language == 'eng':  # English (ISO 639-3: eng)
            return self.preprocess_english(text)
        elif self.language == 'fra':  # French (ISO 639-3: fra)
            return self.preprocess_french(text)
        elif self.language == 'ita':  # Italian (ISO 639-3: ita)
            return self.preprocess_italian(text)
        elif self.language == 'rus':  # Russian (ISO 639-3: rus)
            return self.preprocess_russian(text)
        elif self.language == 'tur':  # Turkish (ISO 639-3: tur)
            return self.preprocess_turkish(text)
        elif self.language == 'spa':  # Spanish (ISO 639-3: spa)
            return self.preprocess_spanish(text)
        elif self.language == 'ell':  # Greek (ISO 639-3: ell)
            return self.preprocess_greek(text)
        elif self.language == 'ron':  # Greek (ISO 639-3: ell)
            return self.preprocess_romania(text)


    ### English-specific Preprocessing
    def preprocess_english(self, text):
        # Lowercasing English text
        text = self.__lowercase_text(text)
        # Handling contractions and removing extra spaces
        text = self.__handle_english_contractions(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_romania(self,text):
      # Step 1: Lowercasing
      text = text.lower()

      # Step 2: Whitespace and punctuation cleaning
      text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
      text = re.sub(r'\s+', ' ', text).strip()  # Removes extra whitespaces

      # Step 3: Diacritic normalization (optional, only if you want to remove diacritics)
      text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
      return text

    ### French-specific Preprocessing
    def preprocess_french(self, text):
        # Lowercasing French text
        text = self.__lowercase_text(text)
        # Normalizing accents in French text
        text = self.__normalize_french_accents(text)
        # Handling French punctuation and spacing
        text = self.__normalize_french_punctuation(text)
        text = self.__remove_extra_whitespace(text)
        return text

    ### Italian-specific Preprocessing
    def preprocess_italian(self, text):
        # Lowercasing Italian text
        text = self.__lowercase_text(text)
        # Italian has no specific accents, but punctuation may need normalization
        text = self.__normalize_italian_punctuation(text)
        text = self.__remove_extra_whitespace(text)
        return text

    ### Russian-specific Preprocessing
    def preprocess_russian(self, text):
        # Removing extra whitespace from Russian text
        text = self.__remove_extra_whitespace(text)
        # Handle case sensitivity if needed
        # Note: Lowercasing Russian is optional depending on your task
        return text

    ### Turkish-specific Preprocessing
    def preprocess_turkish(self, text):
        # Lowercasing Turkish text (be cautious with dotted/undotted "i")
        text = self.__lowercase_turkish(text)
        # Handling Turkish-specific punctuation
        text = self.__normalize_turkish_punctuation(text)
        text = self.__remove_extra_whitespace(text)
        return text

    ### Spanish-specific Preprocessing
    def preprocess_spanish(self, text):
        # Lowercasing Spanish text
        text = self.__lowercase_text(text)
        # Handling Spanish accents
        text = self.__normalize_spanish_accents(text)
        # Removing extra whitespace
        text = self.__remove_extra_whitespace(text)
        return text

    ### Greek-specific Preprocessing
    def preprocess_greek(self, text):
        # Lowercasing Greek text
        text = self.__lowercase_text(text)
        # Handling Greek accents and punctuation
        text = self.__normalize_greek_accents(text)
        text = self.__remove_extra_whitespace(text)
        return text

    # Utility Functions for All Languages
    def __remove_extra_whitespace(self, text):
        return " ".join(text.split())

    def __lowercase_text(self, text):
        return text.lower()

    # Language-Specific Utility Functions
    def __handle_english_contractions(self, text):
        contractions = {"I'm": "I am", "you're": "you are", "isn't": "is not", "can't": "cannot"}
        for contraction, expanded in contractions.items():
            text = text.replace(contraction, expanded)
        return text

    def __normalize_french_accents(self, text):

        return unicodedata.normalize('NFC', text)

    def __normalize_french_punctuation(self, text):
        # French punctuation often uses non-breaking spaces before : ; ! ?
        return text.replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!')

    def __normalize_italian_punctuation(self, text):
        # Italian-specific punctuation normalization if necessary
        return text.replace("’", "'")  # Replace curly quotes

    def __lowercase_turkish(self, text):
        # Turkish lowercase with special handling for dotted/undotted "i"
        return text.replace('I', 'ı').replace('İ', 'i').lower()

    def __normalize_turkish_punctuation(self, text):
        # Turkish-specific punctuation normalization
        return text.replace("’", "'")  # Replace curly quotes

    def __normalize_spanish_accents(self, text):
        import unicodedata
        return unicodedata.normalize('NFC', text)

    def __normalize_greek_accents(self, text):
        import unicodedata
        return unicodedata.normalize('NFC', text)





model_dict = {
    "eng": ["patrick844/translation_en_ar","Helsinki-NLP/opus-mt-en-ar"],
    "fra": ["patrick844/translation_fr_ar","Helsinki-NLP/opus-mt-fr-ar"],
    "ita": ["patrick844/translation_it_ar","Helsinki-NLP/opus-mt-it-ar"],
    "ron":["patrick844/translation_ro_en","Helsinki-NLP/opus-mt-roa-en","patrick844/translation_en_ar","Helsinki-NLP/opus-mt-en-ar"],
    "rus":["Helsinki-NLP/opus-mt-ru-ar","Helsinki-NLP/opus-mt-ru-ar"],
    "tur":["Helsinki-NLP/opus-mt-tr-ar","Helsinki-NLP/opus-mt-tr-ar"],
    "spa":["Helsinki-NLP/opus-mt-es-ar","Helsinki-NLP/opus-mt-es-ar"],
    "ell":["Helsinki-NLP/opus-mt-el-ar","Helsinki-NLP/opus-mt-el-ar"]
}

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

def translation(data, model_config, code, max_length=128, num_beams=3):
    """
    Function to load a fine-tuned MarianMT model and perform translation on a DataFrame column with a progress bar.

    Parameters:
    - data: DataFrame containing the texts to be translated.
    - model_name: Path to the fine-tuned model (default is "translator_en_ar").
    - source_column: The name of the column in the DataFrame that contains the text in the source language.
    - max_length: The maximum length of input sequences for translation (default is 128).
    - num_beams: The number of beams for beam search during inference (default is 3).

    Returns:
    - translated_text: A translated texts.
    """


    token= os.getenv('HF_TOKEN')

    # Load the tokenizer and the fine-tuned model
    tokenizer = MarianTokenizer.from_pretrained(model_config[1], token=token)
    model = MarianMTModel.from_pretrained(model_config[0],token=token)

    if len(model_config)>2:
      model_ar = MarianMTModel.from_pretrained(model_config[2],token=token)
      tokenizer_ar = MarianTokenizer.from_pretrained(model_config[3], token=token)
      data = ">>eng<< "+ data



    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Perform translation (inference)
    translated_tokens = model.generate(**inputs, num_beams=num_beams)

    # Decode the generated tokens to human-readable text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


    if len(model_config)>2:
       inputs = tokenizer_ar(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)


       if code == "ron":
        # Perform translation (inference)
        translated_tokens = model_ar.generate(**inputs, num_beams=num_beams)

        # Decode the generated tokens to human-readable text
        translated_text = tokenizer_ar.decode(translated_tokens[0], skip_special_tokens=True)
        print(f"should be arab {translated_text }")

        return translated_text


    return translated_text

def detect_language(text):

        # Detect the language for both columns
        code = fasttext_model.predict(text)[0][0].replace("__label__", "").split("_")[0]
        # Return both language codes (English, Arabic)
        return code



def preprocessing(code, text):
  processeur = TextPreprocessor(code)
  text = processeur.process(text)
  return text


def inference(text):
  code = detect_language(text)
  text = preprocessing(code,text)
  model_config = model_dict[code]
  print("code: ", code)
  print(text)
  text = translation(text,model_config,code)
  language_name = langcodes.get(code).language_name()
  return text,language_name