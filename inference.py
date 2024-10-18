
from lingowiz.utils import lang_code, abbreviation_dict, model_dict
from lingowiz.inference_preprocessing import TextPreprocessor
import string
import pandas as pd
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel,GenerationConfig
from huggingface_hub import hf_hub_download
import unicodedata
import os
import fasttext
from dotenv import load_dotenv
import langcodes
import language_tool_python
import mlflow
import re
import warnings
import torch
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load languages tool
tool = language_tool_python.LanguageTool('ar')
tool_en = language_tool_python.LanguageTool('en')

# Initialize
tqdm.pandas()
load_dotenv()

print("Loading the FastText language detection model...")
print("FastText model loaded.")
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
fasttext_model = fasttext.load_model(model_path)

def detect_language(text):
    # Detect the language for both columns
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    code = fasttext_model.predict(text)[0][0].replace("__label__", "").split("_")[0]
    # Return both language codes (English, Arabic)
    return code



def preprocessing(code, text):
  processeur = TextPreprocessor(code)
  text = processeur.process(text)
  return text

def processing(text,code):

  text = preprocessing(code,text)
 
  print("Finding Model ...")
  model_config = model_dict[code]
  return text, model_config




def chunk_data(chunk_size,text):

    chunk_text=""
    chunk_list = []
    for i in range(0,len(text.split(" ")),chunk_size):
        if i+chunk_size > len(text.split(" ")):
            chunk_text = text.split(" ")[i:]
        else:
            chunk_text = text.split(" ")[i:i+chunk_size]
        chunk_list.append(chunk_text)
    return chunk_list

def compute_rating(tool,translated_text,language_name):


  print("Checking Error ...")
  matches = tool.check(translated_text)
  number_errors = len(matches)
  length_sentence = len(translated_text.split(" "))

  print("Generating translation rating ...")
  rating = 1-(number_errors/length_sentence)

  print(f"Source: {language_name} \n Number of Errors: {number_errors} \n Sentence Length: {length_sentence} \n Rating: {rating}")
  return rating

  
def mlflow_logging(source_language, target_language, original,translation,rating):


  experiment_name=f"translator_{source_language}_{target_language}_spec"
  experiment = mlflow.get_experiment_by_name(name=experiment_name)
  runs=""
  if experiment:
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    run_id  =runs.iloc[0,0]
    mlflow.set_experiment(experiment_name)
  else:
      mlflow.set_experiment(experiment_name)
      run_id=None

  # Start the MLflow run
  with mlflow.start_run(run_id=run_id, nested=True):
      # List artifacts for the given run_id
      artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)

      # Check if 'inference_data' is already logged
      artifact_names = [artifact.path for artifact in artifacts]
      artifact_path = "inference_data.csv"  # Adjust this path if the artifact was stored in a subdirectory


      if artifact_path in artifact_names:

        # Load Existing Atrifact
        local_artifact_path = mlflow.artifacts.download_artifacts(artifact_path=artifact_path, run_id=run_id)
        df = pd.read_csv(local_artifact_path)

        # Add new data to existing inference data
        new_data = [{"Original": original, "Translation": translation, "Source":source_language, 'Rating': rating}]
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df,new_df])
        df.to_csv("inference_data.csv", index=False)

        #Log new inference data on mlflow
        mlflow.log_artifact("inference_data.csv")
        print("Addindg data to inference_data.csv")

      else:
          # Define the data for the DataFrame
          df = [{"Original": original, "Translation": translation, "Source":source_language, 'Rating': rating}]
          new_df = pd.DataFrame(df)

          # Save the DataFrame to a CSV file
          new_df.to_csv("inference_data.csv", index=False)

          # Log the CSV file as an artifact
          mlflow.log_artifact("inference_data.csv")
          print("Creating new DF and logging as artifact")


def translation(data, model_config,language_name="", max_length=512, num_beams=3):
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

    # HuggingFace token
    token= os.getenv('HF_TOKEN')

    # Load the tokenizer and the fine-tuned model
    tokenizer = MarianTokenizer.from_pretrained(model_config[1], token=token)
    model = MarianMTModel.from_pretrained(model_config[0],token=token)

    if len(model_config)>2:

      # Initialize additional model and tokenizer (English - Arabic)
      model_ar = MarianMTModel.from_pretrained(model_config[2],token=token)
      tokenizer_ar = MarianTokenizer.from_pretrained(model_config[3], token=token)
      original_text = data

      # Format data for model
      data = ">>eng<< "+ data


    # Tokenize data (tansform text into number, same transformation as training)
    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True,max_length=max_length)
    
    # Perform translation (inference)
    translated_tokens = model.generate(**inputs, num_beams=num_beams, max_length=512)

    # Decode the generated tokens to human-readable text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


    if len(model_config)>2:
       rating = compute_rating(tool_en, translated_text, language_name) # Rating
       print(f"Logging: {language_name} to English ")
       mlflow_logging(language_name, "English",original_text, translated_text,rating) # Logging 
       inputs = tokenizer_ar(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length) # Tokenizing
       translated_tokens = model_ar.generate(**inputs, num_beams=num_beams) # Translationn
       translated_text = tokenizer_ar.decode(translated_tokens[0], skip_special_tokens=True) # Decode the generated tokens to human-readable text
       return translated_text
    else:
      return translated_text

def inference(text):
  
  # Detecting Language
  print("Detecting Language ...")
  code = detect_language(text)
  if code in list(model_dict.keys()):
    code = code
  else:
    code="eng"
  language_name = langcodes.get(code).language_name()
  # Checking Text Length
  chunk_size=100
  translated_text = ""
  if len(text.split(" ")) > chunk_size:
        chunk_list = chunk_data(chunk_size,text) # Chunking data if text too big

        for chunk in chunk_list:

          chunk = " ".join(chunk)

          print("Processing Text...")
          processed_text, model_config = processing(chunk,code) # Processing

          print("Translating ...")
          translated_text+=translation(processed_text,model_config,language_name) # Translating
          print("")
  else:
        print("Processing Text ...")
        processed_text, model_config = processing(text,code) # Processing

        print("Translating ...")
        translated_text = translation(processed_text,model_config,language_name)
        print("")
  if language_name=="Romanian":
     language_name="English"

  rating = compute_rating(tool, translated_text, language_name)

  print("Translation Complete")
  print("Logging ...")
  mlflow_logging(language_name, "Arabic", text,translated_text,rating)
  print("Logging Complete")
  return translated_text,language_name,rating
  