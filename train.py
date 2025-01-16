from pickle import TRUE
# Importing Librairies
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datasets import Dataset
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments,GenerationConfig, TrainerCallback # HF librairies for fine tuning
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
os.environ["WANDB_MODE"] = "disabled"
load_dotenv()
token= os.getenv('HF_TOKEN')
tqdm.pandas()
mlflow.autolog(disable=True)
mlflow.transformers.autolog(disable=True)
mlflow.pytorch.autolog(disable=True)

def split_test(df, test_size=0.2, random_state=None):
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

def split_eval(dataset, eval_size=0.2, random_state=42):
    """
    Split a dataset into training and evaluation sets.

    Parameters:
    - dataset: The dataset to split (e.g., a list, pandas DataFrame, or numpy array).
    - eval_size: The proportion of the dataset to include in the evaluation set (default is 20%).
    - random_state: Controls the shuffling applied to the data before splitting (default is 42 for reproducibility).

    Returns:
    - train_data: The portion of the dataset to be used for training.
    - eval_data: The portion of the dataset to be used for evaluation.
    """
    train_data, eval_data = train_test_split(dataset, test_size=eval_size, random_state=random_state)
    return train_data, eval_data


class EmptyCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Clearing cache after epoch {state.epoch}...")
        torch.cuda.empty_cache()

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
        callbacks=[EmptyCacheCallback()]

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




def initialize(data_train, data_eval,base_model,src,trg,unfreeze_layer=0):

    # Load the tokenizer from the pre-trained MarianMT model
    print("")
    print("Intializing Tokenizer ...")
    tokenizer = MarianTokenizer.from_pretrained(base_model, token=token )

    print("Initializing Model ...")
    model = MarianMTModel.from_pretrained(base_model, token=token)



    print("Freezing embedding layer")
    model.model.shared.requires_grad = False

    # Unfreeze encoder layer for source language encoding
    print(f"Freezing last layers of encoder ({unfreeze_layer}) ...")
    for layer in model.model.encoder.layers[:-unfreeze_layer]:
      for param in layer.parameters():
        param.requires_grad = False  # Freeze the encoder layers

    # Unfreeze decoder layer for target language generation
    print(f"UnFreezing last layers of decoder ({unfreeze_layer}) ...")
    for layer in model.model.decoder.layers[:-unfreeze_layer]:
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




def training_pipeline(df,src,base_model,steps,
                      batch_size,learning_rate,epochs, warmup, special_model=None,trg_language="Arabic",layer=0):


    output_path = f"translator_{src}_Arabic_spec"
    print("Data Split train - eval")
    df_train, df_eval = split_eval(df,0.2)
    data_train = Dataset.from_pandas(df_train)
    data_eval = Dataset.from_pandas(df_eval)
    print("")
    print(f"Length train data {len(df_train)} \nLength eval data {len(df_eval)}")


    #  Initialing model, tokenizer, data
    tokenized_datasets_train,tokenized_datasets_eval, model, tokenizer = initialize(data_train,data_eval,base_model,src,trg_language,layer)
    print("")
    trainer, train_metrics = train(model,tokenized_datasets_train,tokenized_datasets_eval,steps,batch_size,learning_rate,epochs, warmup,tokenizer, False)
    print("Training Complete")
    log_history = trainer.state.log_history

    trainer.model.save_pretrained(output_path,safe_serialization=False)
    tokenizer.save_pretrained(output_path)

    # Upload to hugging face
    print("Uploading model to hugging face ...")
    api = HfApi(token=token)
    api.upload_folder(folder_path=output_path,repo_id=f"patrick844/{output_path}",token=token)

    # Logging
    experiment_name = output_path
    log_params(base_model, steps, batch_size, learning_rate, epochs, warmup_steps,experiment_name,df,model,tokenizer,log_history)
