from lingowiz.utils import abbreviation_dict
import unicodedata
import re


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
        else:
          return self.preprocess_english(text)


    ### English-specific Preprocessing
    def preprocess_english(self, text):
        # Lowercasing English text
        text = self.__process_medical_data(text)
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
    def __process_medical_data(self,data):
      data = data.replace("."," ")
      data = data.replace("=", " ")
      data = data.replace(".."," ")
      data = data.replace("..."," ")
      data = data.lower()
      data = self.__expand_abbreviations(data,abbreviation_dict)
      data = self.__add_space_between_letters_and_numbers(data)
      return data

    def __add_space_between_letters_and_numbers(self,text):
    # Use regex to insert a space between letters and numbers
      separated_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
      return separated_text

    # Function to expand abbreviations using the dictionary
    def __expand_abbreviations(self,text, abbreviation_dict):
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviation_dict.keys()) + r')\b')
        expanded_text = pattern.sub(lambda x: abbreviation_dict[x.group()], text)
        return expanded_text

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



