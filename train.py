from pickle import TRUE
# Importing Librairies
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datasets import Dataset
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments,GenerationConfig # HF librairies for fine tuning
import evaluate
from dotenv import load_dotenv
from lingowiz.metrics import evaluation
import mlflow
from datetime import datetime
from mlflow.data.pandas_dataset import PandasDataset
import shutil
from huggingface_hub import HfApi
import json
import requests
from google.colab import userdata
import torch
import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
token= os.getenv('HF_TOKEN')
tqdm.pandas()
bool_model = False

def split_dataframe(df, test_size=0.2, random_state=None):
    """
    Splits the dataframe into training and testing sets based on the given percentage.

    Parameters:
    - df: DataFrame to split
    - test_size: Fraction of the data to reserve for testing (default: 0.2)
    - random_state: Seed used by the random number generator (optional)

    Returns:
    - df_train: Training set DataFrame
    - df_test: Testing set DataFrame
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test


def load_models():
  bleu_m = evaluate.load("bleu")
  bertscore_m = evaluate.load("bertscore")
  return True,bleu_m,bertscore_m


def compute_metrics(eval_preds, tokenizer,bool_model):
    preds, labels = eval_preds
    torch.cuda.empty_cache()
    if bool_model ==False:
      bool_model, bleu_m,bertscore_m = load_models()

    # If predictions are logits, convert them to token IDs using argmax
    if isinstance(preds, tuple):
        preds = preds[0]  # Extract logits if they are in a tuple

    # Apply argmax to get the most likely token IDs
    pred_ids = torch.argmax(torch.tensor(preds), dim=-1)

    # Decode the token IDs into text
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # BLEU expects references to be a list of lists, so each label must be inside a list
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]  # Wrap each label in a list

    # Compute BLEU score
    bleu = bleu_m.compute(predictions=decoded_preds, references=decoded_labels)

    # Compute BERTScore (note: BERTScore expects the language to be specified)
    bert = bertscore_m.compute(predictions=decoded_preds, references=decoded_labels, lang="en")["precision"]

    # Return all metrics
    return {
        "bleu": bleu["bleu"],
        "bert": bert[0]  # Return the F1 score from BERTScore
    }


def train(model, tokenized_datasets_train, tokenized_datasets_eval,steps,
          batch_size,lr,epochs, warmup, tokenizer,bool_model):
    # Adjust training arguments for small dataset
    
    print("")
    print("Initializaing Training Arguments ...")
    training_args = TrainingArguments(
        output_dir="temp",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation
        learning_rate=lr,                        # Learning rate for fine-tuning
        num_train_epochs=epochs,                 # Number of epochs
        warmup_steps=warmup,
        fp16=True,                               # Use mixed precision if on GPU
    )

    trainer = Trainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_eval,
        compute_metrics=lambda p: compute_metrics(p, tokenizer,bool_model)

    )
        



    # Train the model
    print("Training ...")
    train_metrics = trainer.train()

    # Return the trainer
    return trainer, train_metrics

# Tokenization function for both source (English) and target (Arabic), transforming sentences into list of words
def tokenize_function(data,tokenizer,src,trg):

    # Tokenize the source (English) text
    source = tokenizer(data[src], truncation=True, padding="max_length", max_length=512)

    # Tokenize the target (Arabic) text
    targets = tokenizer(data[trg], truncation=True, padding="max_length", max_length=512)

    # Set the 'labels' field to the tokenized target (Arabic) text
    source["labels"] = targets["input_ids"]

    return source

def split_data(dataset, test_size=0.2, random_state=42):
    """
    Split a dataset into training and evaluation sets.

    Parameters:
    - dataset: The dataset to split (e.g., a list, pandas DataFrame, or numpy array).
    - test_size: The proportion of the dataset to include in the evaluation set (default is 20%).
    - random_state: Controls the shuffling applied to the data before splitting (default is 42 for reproducibility).

    Returns:
    - train_data: The portion of the dataset to be used for training.
    - eval_data: The portion of the dataset to be used for evaluation.
    """
    train_data, eval_data = train_test_split(dataset, test_size=test_size, random_state=random_state)
    return train_data, eval_data


def initialize(data_train, data_eval,t_type, special_model,base_model,src,trg,layer=0):

    # Load the tokenizer from the pre-trained MarianMT model
    print("")
    print("Intializing Tokenizer ...")
    tokenizer = MarianTokenizer.from_pretrained(base_model )

    print("Initializing Model ...")
    model = MarianMTModel.from_pretrained(base_model)

    
    print("Freezing embedding layer")
    model.model.shared.requires_grad = False

    print(f"Freezing last {layer} of encoder ...")
    for layer in model.model.encoder.layers[:-layer]:
      for param in layer.parameters():
        param.requires_grad = False  # Freeze the encoder layers

    print("Freezing last 2 layers of decoder ...")
    for param in model.model.decoder.layers[:-2]:
      for param in layer.parameters():
        param.requires_grad = False

    # Apply tokenization to the train dataset in batches (batched=True) for efficiency
    tokenized_datasets_train = data_train.map(tokenize_function, batched=True,fn_kwargs={"tokenizer": tokenizer, "src":src,"trg":trg})
    tokenized_datasets_eval = data_eval.map(tokenize_function, batched=True,fn_kwargs={"tokenizer": tokenizer, "src":src,"trg":trg})
    print("Initialization Complete")
    # Return the tokenized dataset, the model, and the tokenizer for further use
    return tokenized_datasets_train,tokenized_datasets_eval, model, tokenizer



def log_params(base_model, steps, batch_size, learning_rate, epochs, warmup_steps,experiment_name,df,model,tokenizer,log_history):
      """
    Logs parameters to the current MLflow run.

    Parameters:
    - base_model: The base model used for training (e.g., 'Helsinki-NLP/opus-mt-en-ar')
    - steps: Total training steps
    - batch_size: Batch size used during training
    - learning_rate: Learning rate used for training
    - epochs: Number of epochs for training
    - warmup_steps: Number of warmup steps for the learning rate schedule
    """
      now = datetime.now()
      formatted_date_time = now.strftime("%Y-%m-%d_%H:%M")
      # Create or set the experiment

      mlflow.set_experiment(experiment_name)
      run_name = experiment_name+"_"+str(formatted_date_time)
      with mlflow.start_run(run_name=run_name,nested=True):

          print("Logging Hyperparameters")
          mlflow.log_param("base_model", base_model)
          mlflow.log_param("batch_size", batch_size)
          mlflow.log_param("learning_rate", learning_rate)
          mlflow.log_param("warmup_steps", warmup_steps)


          print("Logging Data")
          dataset: PandasDataset = mlflow.data.from_pandas(df, source="dataset_train")
          mlflow.log_input(dataset,context="train")

          print("Logging Model Metrics and Params")
          for params in log_history[-2:]:
            for key, value in params.items():
              mlflow.log_param(key,value)

          print("Logging Model...")
          body = {
              "model_name":experiment_name,
              "run_name": run_name
          }
          requests.post("https://ideal-amoeba-specially.ngrok-free.app/mlflow",json=body, verify=False)

          print("Logging Complete")




def training_pipeline(df,t_type,src,base_model,steps,
                      batch_size,learning_rate,epochs, warmup, special_model=None,trg_language="Arabic",layer=0):

    # Disable unnecessary logging
    # mlflow.autolog(disable=True)
    # mlflow.transformers.autolog(disable=True)
    # mlflow.pytorch.autolog(disable=True)

    bool_model=False

    output_path = f"translator_{src}_Arabic_spec"

    df_train, df_eval = split_data(df,0.1)
    data_train = Dataset.from_pandas(df_train)
    data_eval = Dataset.from_pandas(df_eval)
    print("")
    print("Data Split")
    print(f"Length train data {len(df_train)} \nLength eval data {len(df_eval)}")


    #  Initialing model, tokenizer, data
    tokenized_datasets_train,tokenized_datasets_eval, model, tokenizer = initialize(data_train,data_eval,t_type,special_model,base_model,src,trg_language,layer)
    print("")
    trainer, train_metrics = train(model,tokenized_datasets_train,tokenized_datasets_eval,steps,batch_size,learning_rate,epochs, warmup,tokenizer, bool_model)
    print("Training Complete")

    log_history = trainer.state.log_history

    trainer.model.save_pretrained(output_path,safe_serialization=False)
    tokenizer.save_pretrained(output_path)
    token = userdata.get('HF_TOKEN')
    api = HfApi(token=token)
    api.upload_folder(folder_path=output_path,repo_id=f"patrick844/{output_path}",token=token)
    experiment_name = output_path
    log_params(base_model, steps, batch_size, learning_rate, epochs, warmup,experiment_name,df_train, trainer.model,tokenizer,log_history)
