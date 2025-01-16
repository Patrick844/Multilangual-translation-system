


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
indirect_translation_language = ["ron"]

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load languages tool
tool = language_tool_python.LanguageTool('ar')
tool_en = language_tool_python.LanguageTool('en')

tools = [tool_en,tool]

# Environment Variables
token= os.getenv('HF_TOKEN')

# Initialize
tqdm.pandas()
load_dotenv()



print("Loading the FastText language detection model...")


def detect_language(text):
    print("FastText model loaded.")
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    fasttext_model = fasttext.load_model(model_path)
    # Detect the language for both columns
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    code = fasttext_model.predict(text)[0][0].replace("__label__", "").split("_")[0]

    # If code not found, default is english
    if code in list(model_dict.keys()):
      code = code
    else:
      code="eng"

    # Return both language codes (English, Arabic)
    return code


def processing(text,code):

  # processeur = TextPreprocessor(code)
  # text = processeur.process(text)

  print("Finding Model ...")
  model_config = model_dict[code]
  return text, model_config

def compute_confidence(scores):
  total_confidence=0
  for step_logits in scores:
    probabilities = torch.softmax(step_logits, dim=-1)  # Convert logits to probabilities
    max_prob, _ = torch.max(probabilities, dim=-1)  # Get the max probability for each token
    total_confidence += torch.mean(max_prob).item()  # Average max probability per step

  # Normalize confidence score
  avg_confidence = total_confidence / len(scores)

  print(f"Confidence Score: {avg_confidence:.2f}")
  return avg_confidence


def compute_rating(tool, translated_text, total_confidence):
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

    print(f"\nNumber of Errors: {number_errors} \nSentence Length: {length_sentence} \nRating: {rating_scaled}")
    return rating_scaled


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

  # New data to be added
  new_data = [{"Original": original, "Translation": translation, "Source":source_language, 'Rating': rating}]

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
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df,new_df])

        #Log new inference data on mlflow
        print("Addindg data to inference_data.csv")

      else:
          # Define the data for the DataFrame
          df = pd.DataFrame(new_data)

          # Log the CSV file as an artifact
          print("Creating new DF and logging as artifact")

      # Save the DataFrame to a CSV file
      df.to_csv("inference_data.csv", index=False)
      mlflow.log_artifact("inference_data.csv")

def generating_translation(tokenizer, model, data,num_beams=5, max_length=512):

    # Tokenize data (tansform text into number, same transformation as training)
    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)

    # Perform translation (inference)
    translated_tokens = model.generate(**inputs,return_dict_in_generate=True,output_scores=True)
    scores = translated_tokens.scores
    confidence = compute_confidence(scores)
    translated_tokens = translated_tokens.sequences
    print(f"translated_sequence from inference     {translated_tokens}")

    # Decode the generated tokens to human-readable text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text,confidence


def initializing_model(model_name):
  model = MarianMTModel.from_pretrained(model_name,token=token)
  return model


def initializing_tokenizer(tokenizer_name):
  tokenizer = MarianTokenizer.from_pretrained(tokenizer_name, token=token)
  return tokenizer

def indirect_translation(data, model_config,translation_list,confidence_list, max_length=512, num_beams=5):

  print("")
  print("Indirect ...")
  print("")
  translation_list, confidence_en = direct_translation(data,model_config,translation_list, confidence_list)



  # Initialize additional model and tokenizer (English - Arabic)
  tokenizer_ar = initializing_tokenizer(model_config[2])
  model_ar = initializing_model(model_config[3])

  translated_text, confidence_ar = generating_translation(tokenizer_ar, model_ar, translation_list[0])
  confidence_list.append(confidence_ar)


  if len(translation_list) == 1:
    translation_list.append(translated_text)
  else:
    translation_list[1] = translation_list[1] + translated_text

  return translation_list,confidence_list


def direct_translation(data, model_config, translation_list, confidence_list, max_length=512, num_beams=5):
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

    # Preserve Original text
    original_text = data

    # Load the tokenizer and the fine-tuned model
    tokenizer = initializing_tokenizer(model_config[1])
    model = initializing_model(model_config[0])

    if len(model_config)>2:
      # Format data for model
      data = ">>eng<< "+ data

    print("Translating ...")
    translated_text,confidence = generating_translation(tokenizer, model, data)
    confidence_list.append(confidence)


    if not translation_list:
        translation_list.append(translated_text)
    else:
        translation_list[0] = translation_list[0] + translated_text
    return translation_list, confidence_list


def chunk_data(chunk_size,text):

    chunk_text=""
    chunk_list = []
    for i in range(0,len(text.split(" ")),chunk_size):
        if i+chunk_size > len(text.split(" ")):
            chunk_text = text.split(" ")[i:]
        else:
            chunk_text = text.split(" ")[i:i+chunk_size]
        chunk_list.append(" ".join(chunk_text))
    return chunk_list


def process_chunk(chunk_list,code, translation_func, translation_list, confidence_list):
    translated_text=""
    for chunk in chunk_list:

      print("Processing Text...")
      processed_text, model_config = processing(chunk,code) # Processing
      translation_list, confidence_list = translation_func(processed_text,model_config,translation_list,confidence_list) # Translating
      print(translation_list)
      print("")

    return translation_list, confidence_list


def inference(text):

  # Initializing lists
  source_list = []
  translation_list = []
  confidence_list = []
  confidence_en = []
  confidence_ar = []
  rating_list = []

  confdience = 0

  # Detecting Language
  print("Detecting Language ...")
  code = detect_language(text)
  source_language = langcodes.get(code).language_name()



  chunk_size=100
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
          chunk_list = chunk_data(chunk_size,text) # Chunking data if text too big
          translation_list,confidence_list = process_chunk(chunk_list, code,indirect_translation, translation_list, confidence_list) # process chunk data
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
          processed_text, model_config = processing(text,code) # Processing
          translation_list, confidence_list = indirect_translation(processed_text,model_config, translation_list,confidence_list)



      source_language = source_list[1]

  #Translation 2 - English - Arabic
  else:
    print(f"Direct Translation {source_language} to Arabic")
    print("")
    if not source_list:
        source_list.append(source_language)

    # Direct translation
    # Translation Source - Arabic
    source_language = source_list[0]

    if len(text.split(" ")) > chunk_size:
      chunk_list = chunk_data(chunk_size,text) # Chunking data if text too big
      translation_list,confidence_list = process_chunk(chunk_list, code,direct_translation ,translation_list,confidence_list) # process chunk data
      confidence = sum(confidence_list) / len(confidence_list)
      confidence_list = []
      confidence_list.append(confidence)

    else:
      print("Processing Text ...")
      processed_text, model_config = processing(text,code)
      translation_list, confidence_list = direct_translation(processed_text,model_config, translation_list, confidence_list)


  if len(confidence_list) == 2:
    for i in range(2):
      translated_text = translation_list[i]
      confidence = confidence_list[i]
      tool = tools[i]
      rating = compute_rating(tool, translated_text,confidence)
      rating_list.append(rating)
      source_language=source_list[i]
      if i ==1:
        text = translation_list[0]
        target = "Arabic"
      else:
        target="English"
      print("Logging ...")
      mlflow_logging(source_language, target,text, translated_text,rating)

  else:
      translated_text = translation_list[0]
      confidence = confidence_list[0]
      tool = tools[0]
      sourcce_languahe = source_list[0]
      rating = compute_rating(tool, translated_text,confidence)
      rating_list.append(rating)

      print("Logging ...")
      mlflow_logging(source_language, "Arabic",text, translated_text,rating)
      print("Logging Complete")


  return translation_list, source_list, rating_list


