"""
Module: train.py

This module provides a complete pipeline for fine-tuning machine translation models
using Hugging Face's ``transformers`` library and managing training experiments with MLflow.
It supports data preprocessing, tokenization, model training, and logging results to Hugging Face
and MLflow.

Features:
---------
1. **Data Splitting**:
    - Splits data into training and evaluation sets.

2. **Model Training**:
    - Fine-tunes a MarianMT model on custom datasets.
    - Supports freezing encoder and decoder layers for efficient fine-tuning.

3. **Tokenization**:
    - Tokenizes input and target sentences for training and evaluation.

4. **MLflow Integration**:
    - Logs training hyperparameters, metrics, and models.
    - Automatically creates or updates MLflow experiments.

5. **Hugging Face Integration**:
    - Saves and uploads trained models to the Hugging Face Model Hub.

6. **Callbacks**:
    - Includes a callback to clear GPU memory after each epoch.

Dependencies:
-------------
- ``transformers``: For Hugging Face models and training utilities.
- ``mlflow``: For experiment tracking and logging.
- ``pandas``: For data manipulation.
- ``datasets``: For handling datasets in Hugging Face format.
- ``torch``: For GPU/CPU compatibility during training.
- ``tqdm``: For progress visualization.
- ``requests``: For making API calls to external services.

Example Usage:
--------------
This module is designed for training machine translation models on custom datasets, integrating
key utilities for tokenization, training, logging, and uploading models.

.. code-block:: python

    from translation_training_pipeline import training_pipeline

    df = pd.read_csv("translation_data.csv")
    training_pipeline(
        df=df,
        src="English",
        base_model="Helsinki-NLP/opus-mt-en-ar",
        steps=1000,
        batch_size=16,
        learning_rate=5e-5,
        epochs=3,
        warmup=100,
        trg_language="Arabic",
        layer=1
    )
"""

# Importing Librairies
import os
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
# HF librairies for fine tuning
from transformers import (MarianTokenizer,
                          MarianMTModel,
                          Trainer,
                          TrainingArguments,
                          TrainerCallback)
from dotenv import load_dotenv
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from huggingface_hub import HfApi
import requests
import torch


# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["WANDB_MODE"] = "disabled"
load_dotenv()
token = os.getenv('HF_TOKEN')
tqdm.pandas()
mlflow.autolog(disable=True)
mlflow.transformers.autolog(disable=True)
mlflow.pytorch.autolog(disable=True)


def split_test(df, test_size=0.2, random_state=None):
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        test_size (float, optional): Fraction of the data to reserve for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Training set.
            - pd.DataFrame: Testing set.
    """
    df_train, df_test = train_test_split(df,
                                         test_size=test_size,
                                         random_state=random_state)
    return df_train, df_test


class EmptyCacheCallback(TrainerCallback):
    """
    A callback to clear GPU memory after each training epoch to prevent memory overflow.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Clears the GPU cache after the end of an epoch.

        Args:
            args (TrainingArguments): Training arguments.
            state (TrainerState): Trainer state.
            control (TrainerControl): Trainer control object.
        """
        print(f"Clearing cache after epoch {state.epoch}...")
        torch.cuda.empty_cache()


def train(model,
          tokenized_datasets_train,
          tokenized_datasets_eval,
          batch_size,
          lr,
          epochs,
          warmup,
          tokenizer):
    # Adjust training arguments for small dataset

    """
    Trains a MarianMT model on tokenized datasets.

    Args:
        model (MarianMTModel): The MarianMT model to fine-tune.
        tokenized_datasets_train (Dataset): Tokenized training dataset.
        tokenized_datasets_eval (Dataset): Tokenized evaluation dataset.
        steps (int): Total training steps.
        batch_size (int): Batch size for training and evaluation.
        lr (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
        warmup (int): Number of warmup steps for learning rate scheduling.
        tokenizer (MarianTokenizer): Tokenizer for the MarianMT model.
        bool_model (bool): Reserved for additional configuration (unused in this function).

    Returns:
        tuple: A tuple containing:
            - Trainer: Hugging Face Trainer object after training.
            - TrainOutput: Training metrics.
    """

    print("")
    print("Initializaing Training Arguments ...")

    # Directory to save training checkpoints and outputs
    output_dir = "temp"

    # Evaluate the model at the end of each epoch
    evaluation_strategy = "epoch"

    # Log training metrics at the end of each epoch
    logging_strategy = "epoch"

    # Batch size for training per device (GPU/CPU)
    per_device_train_batch_size = batch_size

    # Batch size for evaluation per device (GPU/CPU)
    per_device_eval_batch_size = batch_size

    # Learning rate for the optimizer during fine-tuning
    learning_rate = lr

    # Number of epochs for training
    num_train_epochs = epochs

    # Number of steps for learning rate warmup
    warmup_steps = warmup

    # Use mixed precision (16-bit floating point) for faster training
    fp16 = True

    # Initialize training arguments for the Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        logging_strategy=logging_strategy,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_eval,
        callbacks=[EmptyCacheCallback()]
    )

    # Train the model
    print("Training ...")
    trainer.train()

    # Return the trainer
    return trainer


# Tokenization function for both source (English) and target (Arabic)
# transforming sentences into list of words
def tokenize_function(data, tokenizer, src, trg):
    """
    Tokenizes source and target texts for training and evaluation.

    Args:
        data (Dataset): The dataset containing source and target texts.
        tokenizer (MarianTokenizer): Tokenizer for the MarianMT model.
        src (str): Name of the source language column in the dataset.
        trg (str): Name of the target language column in the dataset.

    Returns:
        dict: Tokenized input and target sequences, including labels.
    """

    # Tokenize the source (English) text
    source = tokenizer(data[src],
                       truncation=True,
                       padding="max_length",
                       max_length=512)

    # Tokenize the target (Arabic) text
    targets = tokenizer(data[trg],
                        truncation=True,
                        padding="max_length",
                        max_length=512)

    # Set the 'labels' field to the tokenized target (Arabic) text
    source["labels"] = targets["input_ids"]

    return source


def split_eval(dataset, eval_size=0.2, random_state=42):
    """
    Splits a dataset into training and evaluation subsets.

    Args:
        dataset (Dataset): The dataset to split.
        eval_size (float, optional): Proportion of the dataset for evaluation. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - Dataset: Training subset.
            - Dataset: Evaluation subset.
    """
    train_data, eval_data = train_test_split(dataset,
                                             test_size=eval_size,
                                             random_state=random_state)
    return train_data, eval_data


def initialize(data_train,
               data_eval,
               base_model,
               src,
               trg,
               layer=0):
    """
    Initializes the tokenizer, model, and tokenized datasets for training.

    Args:
        data_train (Dataset): Training dataset.
        data_eval (Dataset): Evaluation dataset.
        special_model (str): Special model configuration (optional).
        base_model (str): Pretrained MarianMT model name or path.
        src (str): Source language column in the dataset.
        trg (str): Target language column in the dataset.
        layer (int, optional): Layer freezing configuration. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - Dataset: Tokenized training dataset.
            - Dataset: Tokenized evaluation dataset.
            - MarianMTModel: Initialized MarianMT model.
            - MarianTokenizer: Initialized tokenizer.
    """
    # Load the tokenizer from the pre-trained MarianMT model
    print("")
    print("Intializing Tokenizer ...")
    tokenizer = MarianTokenizer.from_pretrained(base_model)

    print("Initializing Model ...")
    model = MarianMTModel.from_pretrained(base_model)

    # print("Freezing embedding layer")
    # model.model.shared.requires_grad = False

    # # Unfreeze encoder layer for source language encoding
    # print(f"Freezing last layers of encoder ...")
    # for layer in model.model.encoder.layers[:-1]:
    #   for param in layer.parameters():
    #     param.requires_grad = False  # Freeze the encoder layers

    # Unfreeze decoder layer for target language generation
    print("Freezing last layers of decoder ...")
    for layer in model.model.decoder.layers[:-2]:
        for param in layer.parameters():
            param.requires_grad = False

    tokenizer_args = {"tokenizer": tokenizer,
                      "src": src,
                      "trg": trg}
    # Apply tokenization to the train dataset in batches (batched=True)
    tokenized_datasets_train = data_train.map(tokenize_function,
                                              batched=True,
                                              fn_kwargs=tokenizer_args)
    tokenized_datasets_eval = data_eval.map(tokenize_function,
                                            batched=True,
                                            fn_kwargs=tokenizer_args)
    print("Initialization Complete")

    # Return the tokenized dataset, the model,
    # and the tokenizer for further use
    return (tokenized_datasets_train,
            tokenized_datasets_eval,
            model,
            tokenizer)


def log_params(base_model,
               batch_size,
               learning_rate,
               epochs,
               warmup_steps,
               experiment_name,
               df,
               log_history):
    """
    Logs training hyperparameters, data, and model metadata to MLflow.

    Args:
        base_model (str): Base MarianMT model used for training.
        steps (int): Total training steps.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate used during training.
        epochs (int): Number of training epochs.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        experiment_name (str): Name of the MLflow experiment.
        df (pd.DataFrame): Training dataset.
        model (MarianMTModel): Trained MarianMT model.
        tokenizer (MarianTokenizer): Tokenizer used for training.
        log_history (list): Training logs for parameter logging.

    Returns:
        None
    """
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d_%H:%M")
    # Create or set the experiment

    mlflow.set_experiment(experiment_name)
    run_name = experiment_name+"_"+str(formatted_date_time)
    with mlflow.start_run(run_name=run_name,
                          nested=True):

        print("Logging Hyperparameters")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("warmup_steps", warmup_steps)

        print("Logging Data")
        dataset: PandasDataset = mlflow.data.from_pandas(df,
                                                         source="dataset_train")
        mlflow.log_input(dataset, context="train")

        print("Logging Model Metrics and Params")
        for params in log_history[-2:]:
            for key, value in params.items():
                mlflow.log_param(key, value)

        print("Logging Model...")
        body = {
            "model_name": experiment_name,
            "run_name": run_name
        }
        requests.post("https://ideal-amoeba-specially.ngrok-free.app/mlflow",
                      json=body,
                      verify=False)
        print("Logging Complete")


def training_pipeline(df,
                      src,
                      base_model,
                      steps,
                      batch_size,
                      learning_rate,
                      epochs,
                      warmup,
                      special_model=None,
                      trg_language="Arabic",
                      layer=0):

    """
    Executes the end-to-end training pipeline for fine-tuning a MarianMT model.

    Args:
        df (pd.DataFrame): Input dataset for training and evaluation.
        src (str): Source language column name in the dataset.
        base_model (str): Pretrained MarianMT model name or path.
        steps (int): Total training steps.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for fine-tuning.
        epochs (int): Number of training epochs.
        warmup (int): Number of warmup steps for learning rate scheduling.
        special_model (str, optional): Custom model configuration. Defaults to None.
        trg_language (str, optional): Target language name. Defaults to "Arabic".
        layer (int, optional): Layer freezing configuration. Defaults to 0.

    Returns:
        None
    """
    output_path = f"translator_{src}_Arabic_spec"
    print("Data Split train - eval")
    df_train, df_eval = split_eval(df, 0.1)
    data_train = Dataset.from_pandas(df_train)
    data_eval = Dataset.from_pandas(df_eval)
    print("")
    print(f"Length train data {len(df_train)} \nLength eval data {len(df_eval)}")


#  Initialing model, tokenizer, data
    (tokenized_datasets_train,
     tokenized_datasets_eval,
     model,
     tokenizer) = initialize(data_train,
                             data_eval,
                             special_model,
                             base_model,
                             src,
                             trg_language,
                             layer)

    print("")
    trainer = train(
        model,
        tokenized_datasets_train,
        tokenized_datasets_eval,
        steps,
        batch_size,
        learning_rate,
        epochs,
        warmup,
        tokenizer,
        False,
    )

    print("Training Complete")

    log_history = trainer.state.log_history

    trainer.model.save_pretrained(output_path,
                                  safe_serialization=False)
    tokenizer.save_pretrained(output_path)

    # Upload to hugging face
    print("Uploading model to hugging face ...")
    api = HfApi(token=token)
    api.upload_folder(folder_path=output_path,
                      repo_id=f"patrick844/{output_path}",
                      token=token)

    # Logging
    experiment_name = output_path
    log_params(
        base_model,
        steps,
        batch_size,
        learning_rate,
        epochs, warmup,
        experiment_name,
        df_train,
        trainer.model,
        tokenizer,
        log_history)
