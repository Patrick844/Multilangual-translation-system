"""
Module: inference_preprocess.py

This module provides functionality for preprocessing text input during inference.
It includes a ``TextPreprocessor`` class with methods for handling language-specific
text normalization, cleaning, and transformation. The preprocessing ensures that
text data is standardized and ready for model inference.

Classes:
--------
- **TextPreprocessor**:
    A class that encapsulates language-specific preprocessing logic, supporting multiple languages.

Methods:
--------
The methods inside ``TextPreprocessor`` include:
    - ``preprocess_english``: Preprocesses English text (e.g., handling contractions).
    - ``preprocess_french``: Handles French-specific accent and punctuation normalization.
    - ``preprocess_italian``: Processes Italian text with punctuation normalization.
    - ``preprocess_russian``: Removes extra whitespace from Russian text.
    - ``preprocess_turkish``: Handles Turkish-specific lowercase and punctuation rules.
    - ``preprocess_spanish``: Normalizes Spanish accents and whitespace.
    - ``preprocess_greek``: Normalizes Greek text, including accents and punctuation.
    - ``preprocess_romania``: Handles Romanian-specific text transformations.

Utilities:
----------
Language-agnostic helper methods include:
    - ``__process_medical_data``: Processes medical data, expanding abbreviations and cleaning text.
    - ``__expand_abbreviations``: Expands abbreviations using a dictionary.
    - ``__add_space_between_letters_and_numbers``: Adds spaces between letters and numbers.
    - ``__remove_extra_whitespace``: Cleans up redundant spaces in text.
    - ``__normalize_*``: Various normalization methods for punctuation and accents.

Usage:
------
This module is intended to standardize text input for natural language processing tasks,
ensuring compatibility with models during inference.

Example:
    .. code-block:: python

        from inference_preprocess import TextPreprocessor

        # Create a preprocessor for English
        preprocessor = TextPreprocessor(language="eng")
        
        # Preprocess some text
        clean_text = preprocessor.process("I'm feeling great! Let's preprocess this text.")
        print(clean_text)

Dependencies:
-------------
- ``re``: Regular expressions for text pattern matching.
- ``unicodedata``: Unicode support for diacritic normalization.
- ``lingowiz.utils.abbreviation_dict``: A dictionary of abbreviations for medical data expansion.

Note:
-----
Extend or customize the ``TextPreprocessor`` class for additional languages or
preprocessing requirements as needed.
"""


from lingowiz.utils import abbreviation_dict
import unicodedata
import re


class TextPreprocessor:
    """
    A text preprocessing class for handling
    language-specific transformations and normalizations.

    Attributes:
        language (str): The ISO 639-3 language code for preprocessing.
    """

    def __init__(self, language):
        """
        Initializes the TextPreprocessor with a specific language.

        Args:
            language (str): The ISO 639-3 language code
            for the text (e.g., ''eng'' for English).
        """
        self.language = language

    def process(self, text):
        """
        Preprocesses text based on the specified language.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        if self.language == 'eng':
            return self.preprocess_english(text)
        elif self.language == 'fra':
            return self.preprocess_french(text)
        elif self.language == 'ita':
            return self.preprocess_italian(text)
        elif self.language == 'rus':
            return self.preprocess_russian(text)
        elif self.language == 'tur':
            return self.preprocess_turkish(text)
        elif self.language == 'spa':
            return self.preprocess_spanish(text)
        elif self.language == 'ell':
            return self.preprocess_greek(text)
        elif self.language == 'ron':
            return self.preprocess_romania(text)
        else:
            return self.preprocess_english(text)

    def preprocess_english(self, text):
        """
        Preprocesses English text with medical data handling,
        contraction normalization, and whitespace cleaning.

        Args:
            text (str): The English text to preprocess.

        Returns:
            str: The preprocessed English text.
        """
        text = self.__process_medical_data(text)
        text = self.__handle_english_contractions(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_romania(self, text):
        """
        Preprocesses Romanian text with lowercasing,
        punctuation cleaning, and diacritic normalization.

        Args:
            text (str): The Romanian text to preprocess.

        Returns:
            str: The preprocessed Romanian text.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text

    def preprocess_french(self, text):
        """
        Preprocesses French text with accent normalization,
        punctuation handling, and whitespace cleaning.

        Args:
            text (str): The French text to preprocess.

        Returns:
            str: The preprocessed French text.
        """
        text = self.__lowercase_text(text)
        text = self.__normalize_french_accents(text)
        text = self.__normalize_french_punctuation(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_italian(self, text):
        """
        Preprocesses Italian text with punctuation
        normalization and whitespace cleaning.

        Args:
            text (str): The Italian text to preprocess.

        Returns:
            str: The preprocessed Italian text.
        """
        text = self.__lowercase_text(text)
        text = self.__normalize_italian_punctuation(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_russian(self, text):
        """
        Preprocesses Russian text by removing extra whitespace.

        Args:
            text (str): The Russian text to preprocess.

        Returns:
            str: The preprocessed Russian text.
        """
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_turkish(self, text):
        """
        Preprocesses Turkish text with lowercase handling
        and punctuation normalization.

        Args:
            text (str): The Turkish text to preprocess.

        Returns:
            str: The preprocessed Turkish text.
        """
        text = self.__lowercase_turkish(text)
        text = self.__normalize_turkish_punctuation(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_spanish(self, text):
        """
        Preprocesses Spanish text with accent normalization
        and whitespace cleaning.

        Args:
            text (str): The Spanish text to preprocess.

        Returns:
            str: The preprocessed Spanish text.
        """
        text = self.__lowercase_text(text)
        text = self.__normalize_spanish_accents(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def preprocess_greek(self, text):
        """
        Preprocesses Greek text with accent normalization
        and whitespace cleaning.

        Args:
            text (str): The Greek text to preprocess.

        Returns:
            str: The preprocessed Greek text.
        """
        text = self.__lowercase_text(text)
        text = self.__normalize_greek_accents(text)
        text = self.__remove_extra_whitespace(text)
        return text

    def __process_medical_data(self, data):
        """
        Processes medical text data by replacing symbols,
        expanding abbreviations,
        and adding spaces between letters and numbers.

        Args:
            data (str): The medical text to process.

        Returns:
            str: The processed medical text.
        """
        data = data.replace(".", " ")
        data = data.replace("=", " ")
        data = data.replace("..", " ")
        data = data.replace("...", " ")
        data = data.lower()
        data = self.__expand_abbreviations(data, abbreviation_dict)
        data = self.__add_space_between_letters_and_numbers(data)
        return data

    def __add_space_between_letters_and_numbers(self, text):
        """
        Adds spaces between letters and numbers in the text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with spaces added between letters and numbers.
        """
        return re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

    def __expand_abbreviations(self, text, abbreviation_dict):
        """
        Expands abbreviations in the text using a dictionary.

        Args:
            text (str): The text containing abbreviations.
            abbreviation_dict (dict): The abbreviation-to-expansion mapping.

        Returns:
            str: The text with expanded abbreviations.
        """
        pattern = re.compile(
                             r'\b(' +
                             '|'.join(re.escape(key) for key in abbreviation_dict.keys()) +
                             r')\b'
                            )

        return pattern.sub(lambda x: abbreviation_dict[x.group()], text)

    def __remove_extra_whitespace(self, text):
        """
        Removes extra whitespace from the text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with extra whitespace removed.
        """
        return " ".join(text.split())

    def __lowercase_text(self, text):
        """
        Converts text to lowercase.

        Args:
            text (str): The input text.

        Returns:
            str: The lowercase text.
        """
        return text.lower()

    def __handle_english_contractions(self, text):
        """
        Expands English contractions in the text.

        Args:
            text (str): The English text with contractions.

        Returns:
            str: The text with expanded contractions.
        """
        contractions = {"I'm": "I am",
                        "im": "i am"
                        "you're": "you are",
                        "isn't": "is not",
                        "can't": "cannot"}
        for contraction, expanded in contractions.items():
            text = text.replace(contraction, expanded)
        return text

    def __normalize_french_accents(self, text):
        """
        Normalizes French accents in the text.

        Args:
            text (str): The French text with accents.

        Returns:
            str: The normalized text.
        """
        return unicodedata.normalize('NFC', text)

    def __normalize_french_punctuation(self, text):
        """
        Normalizes French punctuation marks.

        Args:
            text (str): The French text with punctuation.

        Returns:
            str: The text with normalized punctuation.
        """
        text = text.replace(' :', ':')
        text = text.replace(' ;', ';')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        return text

    def __normalize_italian_punctuation(self, text):
        """
        Normalizes Italian-specific punctuation.

        Args:
            text (str): The Italian text with punctuation.

        Returns:
            str: The text with normalized punctuation.
        """
        return text.replace("’", "'")

    def __lowercase_turkish(self, text):
        """
        Converts Turkish text to lowercase,
        handling special Turkish characters.

        Args:
            text (str): The Turkish text.

        Returns:
            str: The lowercase Turkish text.
        """
        text = text.replace('I', 'ı')
        text = text.replace('İ', 'i')
        text = text.lower()
        return text

    def __normalize_turkish_punctuation(self, text):
        """
        Normalizes Turkish-specific punctuation.

        Args:
            text (str): The Turkish text with punctuation.

        Returns:
            str: The text with normalized punctuation.
        """
        return text.replace("’", "'")

    def __normalize_spanish_accents(self, text):
        """
        Normalizes Spanish accents in the text.

        Args:
            text (str): The Spanish text with accents.

        Returns:
            str: The normalized Spanish text.
        """
        return unicodedata.normalize('NFC', text)

    def __normalize_greek_accents(self, text):
        """
        Normalizes Greek accents in the text.

        Args:
            text (str): The Greek text with accents.

        Returns:
            str: The normalized Greek text.
        """
        return unicodedata.normalize('NFC', text)
