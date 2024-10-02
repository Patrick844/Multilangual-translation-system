# Importing Librairies
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datasets import Dataset
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments,GenerationConfig # HF librairies for fine tuning
import pandas as pd # DataFrame manipulation (csv file)
from google.colab import drive
import evaluate
from dotenv import load_dotenv
from lingowiz.metrics import evaluation

import torch

load_dotenv()
token= os.getenv('HF_TOKEN')
tqdm.pandas()

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

# bleu_m = evaluate.load("bleu")
# chrf_m = evaluate.load("chrf")
# bertscore_m = evaluate.load("bertscore")



def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    
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

    # Compute chrF score
    chrf = chrf_m.compute(predictions=decoded_preds, references=decoded_labels)

    # Compute BERTScore (note: BERTScore expects the language to be specified)
    bert = bertscore_m.compute(predictions=decoded_preds, references=decoded_labels, lang="en")["precision"]

    # Return all metrics
    return {
        "bleu": bleu["bleu"],
        "chrf": chrf["score"],
        "bert": bert[0]  # Return the F1 score from BERTScore
    }


def train(model, tokenized_datasets_train, tokenized_datasets_eval,steps,
          batch_size,lr,epochs, warmup, tokenizer, t_type  ):
    # Adjust training arguments for small dataset

    if t_type=="general":
        training_args = TrainingArguments(
        output_dir="temp",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_strategy="epoch",     # Log training progress at the end of each epoch
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation
        learning_rate=lr,                        # Learning rate for fine-tuning
        num_train_epochs=epochs,                 # Number of epochs
        warmup_steps=warmup,    
        save_total_limit=1,                 # Warmup steps
        fp16=True,                               # Use mixed precision if on GPU
    )


        # Setting up the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets_train,
            eval_dataset=tokenized_datasets_eval

        )
    else:
                training_args = TrainingArguments(
        output_dir="temp",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_strategy="epoch",     # Log training progress at the end of each epoch
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation
        learning_rate=lr,                        # Learning rate for fine-tuning
        num_train_epochs=epochs,                 # Number of epochs
        warmup_steps=warmup,    
        save_total_limit=1,                 # Warmup steps
        fp16=True,                               # Use mixed precision if on GPU
        
    )


        # Setting up the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets_train,
            eval_dataset=tokenized_datasets_eval,
            compute_metrics=lambda p: compute_metrics(p, tokenizer)

        )


    # Train the model
    trainer.train()

    # Return the trainer
    return trainer

# Tokenization function for both source (English) and target (Arabic), transforming sentences into list of words
def tokenize_function(data,tokenizer,src,trg):
    # Tokenize the source (English) text
    source = tokenizer(data[src], truncation=True, padding="max_length", max_length=128)

    # Tokenize the target (Arabic) text
    target = tokenizer(data[trg], truncation=True, padding="max_length", max_length=128)

    # Set the 'labels' field to the tokenized target (Arabic) text
    source["labels"] = target["input_ids"]

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

    
def initialize(data_train, data_eval,t_type, special_model,base_model,src,trg):

    # Load the tokenizer from the pre-trained MarianMT model
    tokenizer = MarianTokenizer.from_pretrained(base_model)
    if t_type == "general":
      model = MarianMTModel.from_pretrained(base_model)
    else:
        model = MarianMTModel.from_pretrained(special_model,token=token)
        for layer in model.model.encoder.layers[-2:]:
          for param in layer.parameters():
            param.requires_grad = False  # Freeze the encoder layers

    # Tokenize the dataset using the previously defined `tokenize_function`
    # Apply tokenization to the train dataset in batches (batched=True) for efficiency
    tokenized_datasets_train = data_train.map(tokenize_function, batched=True,fn_kwargs={"tokenizer": tokenizer, "src":src,"trg":trg})
    tokenized_datasets_eval = data_eval.map(tokenize_function, batched=True,fn_kwargs={"tokenizer": tokenizer, "src":src,"trg":trg})
    
    # Return the tokenized dataset, the model, and the tokenizer for further use
    return tokenized_datasets_train,tokenized_datasets_eval, model, tokenizer


def training_pipeline(df,t_type,src,base_model,steps,
                      batch_size,lr,epochs, warmup, special_model=None,trg_language="Arabic"):
    if t_type == "general":
      df_train, df_test = split_dataframe(df,0.2,42)
      
      if not os.path.isdir("/content/model"):
        df_train, df_eval = split_data(df_train)
        print("After split train/eval")
        print(len(df_train))

        data_train = Dataset.from_pandas(df_train)
        data_eval = Dataset.from_pandas(df_eval) # Convert the pandas DataFrame into a Hugging Face Dataset
        tokenized_datasets_train,tokenized_datasets_eval, model, tokenizer = initialize(data_train,data_eval,t_type,None,base_model,src,trg_language)
        trainer = train(model,tokenized_datasets_train,tokenized_datasets_eval,steps,batch_size,lr,epochs, warmup,tokenizer,t_type)
        trainer.save_model(f"translator_{src}_Arabic")
        evaluation(df_test,model,tokenizer,src)
    else:
      df_train, df_eval = split_data(df,0.1)
      data_train = Dataset.from_pandas(df_train)
      data_eval = Dataset.from_pandas(df_eval)
      tokenized_datasets_train,tokenized_datasets_eval, model, tokenizer = initialize(data_train,data_eval,t_type,special_model,base_model,src,trg_language)
      trainer = train(model,tokenized_datasets_train,tokenized_datasets_eval,steps,batch_size,lr,epochs, warmup,tokenizer)
      trainer.save_model(f"translator_{src}_Arabic_spec")


# steps,batch_size,lr,epochs, warmup = 1000,16,1e-4, 17, 500 base
# steps,batch_size,lr,epochs, warmup = 50,16,5e-5, 3, 10 special

    

    