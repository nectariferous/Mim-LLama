import os
import subprocess
import sys
import torch
from termcolor import colored
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, set_caching_enabled
from huggingface_hub import HfApi, HfFolder, Repository
import logging
import configparser
import warnings

# Suppress the warning about weights not being initialized
warnings.filterwarnings("ignore", message="Some weights of")

# Set cache disabled to avoid slowdowns
set_caching_enabled(False)

# Function to install required packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed
def install_required_packages(config):
    required_packages = config["DEFAULT"]["required_packages"].split(",")
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(colored(f"Installing {package}...", "yellow"))
            install_package(package)

# Define the paths and model names
def get_paths_and_model_names(config):
    output_dir = config["DEFAULT"]["output_dir"]
    mini_model_name = config["DEFAULT"]["mini_model_name"]
    return output_dir, mini_model_name

# Define the dataset parameters
def get_dataset_params(config):
    max_sequence_length = int(config["DEFAULT"]["max_sequence_length"])
    num_classes = int(config["DEFAULT"]["num_classes"])
    return max_sequence_length, num_classes

# Load the dataset
def load_dataset_from_hub():
    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
    train_dataset = dataset["train"]

    # Split the train dataset into train and validation
    train_val_split = train_dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    return train_dataset, val_dataset

# Load a pre-trained model and tokenizer
def load_model_and_tokenizer(model_name, num_classes):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer

# Tokenize the datasets
def tokenize_function(examples, tokenizer, max_sequence_length):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_sequence_length)

# Create a training arguments object
def create_training_args(config):
    output_dir = config["DEFAULT"]["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config["DEFAULT"]["num_train_epochs"]),
        per_device_train_batch_size=int(config["DEFAULT"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["DEFAULT"]["per_device_eval_batch_size"]),
        eval_strategy=config["DEFAULT"]["evaluation_strategy"],
        save_strategy=config["DEFAULT"]["save_strategy"],
        learning_rate=float(config["DEFAULT"]["learning_rate"]),
        save_total_limit=int(config["DEFAULT"]["save_total_limit"]),
        load_best_model_at_end=config["DEFAULT"]["load_best_model_at_end"].lower() == "true",
        metric_for_best_model=config["DEFAULT"]["metric_for_best_model"],
        greater_is_better=config["DEFAULT"]["greater_is_better"].lower() == "true",
        fp16=True,  # Enable mixed-precision training
        report_to="none",  # Disable wandb logging
    )
    return training_args

# Define a compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": (predictions == labels).float().mean().item()}

# Train the model
def train_model(model, tokenized_train, tokenized_val, training_args, compute_metrics):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    trainer.train()

# Save the model and tokenizer
def save_model_and_tokenizer(model, tokenizer, output_dir, mini_model_name):
    model_save_path = os.path.join(output_dir, mini_model_name)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(colored(f"Model saved to {model_save_path}", "green"))

# Upload the model to Hugging Face
def upload_model_to_hugging_face(model_save_path, mini_model_name):
    print(colored("Uploading the model to Hugging Face...", "green"))
    api = HfApi()
    hf_folder = HfFolder()
    repo_name = mini_model_name
    repo_url = api.create_repo(repo_name)
    repo = Repository(local_dir=model_save_path, clone_from=repo_url)
    repo.push_to_hub()
    print(colored(f"Model uploaded to {repo_url}", "green"))

# Main function
def main():
    # Load configuration
    config = configparser.ConfigParser()
    try:
        config.read("config.ini")
    except configparser.ParsingError as e:
        print(colored(f"Error parsing config.ini: {e}", "red"))
        sys.exit(1)

    # Install required packages
    install_required_packages(config)

    # Get paths and model names
    output_dir, mini_model_name = get_paths_and_model_names(config)

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset parameters
    max_sequence_length, num_classes = get_dataset_params(config)

    # Load the dataset
    train_dataset, val_dataset = load_dataset_from_hub()

    # Load a pre-trained model and tokenizer
    model_name = config["DEFAULT"]["model_name"]
    model, tokenizer = load_model_and_tokenizer(model_name, num_classes)

    # Tokenize the datasets using multi-processing
    tokenized_train = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer, max_sequence_length), batched=True, num_proc=os.cpu_count())
    tokenized_val = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer, max_sequence_length), batched=True, num_proc=os.cpu_count())

    # Create a training arguments object
    training_args = create_training_args(config)

    # Train the model
    train_model(model, tokenized_train, tokenized_val, training_args, compute_metrics)

    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer, output_dir, mini_model_name)

    # Upload the model to Hugging Face
    upload_model_to_hugging_face(os.path.join(output_dir, mini_model_name), mini_model_name)

if __name__ == "__main__":
    main()
