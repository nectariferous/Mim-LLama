import os
import subprocess
import sys
import requests
import time
import logging
import configparser
import warnings
from termcolor import colored
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, set_caching_enabled
from huggingface_hub import HfApi, HfFolder, Repository

# Suppress the warning about weights not being initialized
warnings.filterwarnings("ignore", message="Some weights of")

# Set cache disabled to avoid slowdowns
set_caching_enabled(False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mistral API credentials and endpoint
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "ZLtWpfzof2FfNJK3AnZjz3yy0ljFR4J7")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def install_package(package: str) -> None:
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_required_packages(config: configparser.ConfigParser) -> None:
    """Ensure required packages are installed."""
    required_packages = config["DEFAULT"]["required_packages"].split(",")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(install_package, package.strip()) for package in required_packages if not is_package_installed(package.strip())]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Installing packages"):
            future.result()

def is_package_installed(package: str) -> bool:
    """Check if a package is already installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def get_suggestions(terminal_output: str) -> List[Dict[str, Any]]:
    """Get suggestions from Mistral API."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"input": terminal_output}
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json().get("suggestions", [])
    except requests.RequestException as e:
        logger.error(colored(f"Error getting suggestions: {e}", "red"))
        return []

def apply_suggestions(suggestions: List[Dict[str, Any]]) -> None:
    """Apply suggestions to files."""
    for suggestion in suggestions:
        file_path = suggestion.get("file_path")
        line_number = suggestion.get("line_number")
        new_content = suggestion.get("new_content")
        if all([file_path, line_number is not None, new_content]):
            try:
                with open(file_path, "r") as file:
                    lines = file.readlines()
                lines[line_number - 1] = new_content + "\n"
                with open(file_path, "w") as file:
                    file.writelines(lines)
                logger.info(colored(f"Applied suggestion to {file_path} at line {line_number}", "green"))
            except IOError as e:
                logger.error(colored(f"Error applying suggestion to {file_path}: {e}", "red"))

def capture_terminal_output(command: str) -> str:
    """Capture terminal output from a command."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(colored(f"Command timed out: {command}", "yellow"))
        return ""

def get_paths_and_model_names(config: configparser.ConfigParser) -> tuple:
    """Define the paths and model names."""
    output_dir = config["DEFAULT"]["output_dir"]
    mini_model_name = config["DEFAULT"]["mini_model_name"]
    return output_dir, mini_model_name

def get_dataset_params(config: configparser.ConfigParser) -> tuple:
    """Define the dataset parameters."""
    max_sequence_length = int(config["DEFAULT"]["max_sequence_length"])
    num_classes = int(config["DEFAULT"]["num_classes"])
    return max_sequence_length, num_classes

def load_dataset_from_hub():
    """Load the dataset."""
    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
    train_dataset = dataset["train"]

    # Split the train dataset into train and validation
    train_val_split = train_dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    return train_dataset, val_dataset

def load_model_and_tokenizer(model_name: str, num_classes: int) -> tuple:
    """Load a pre-trained model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer

def tokenize_function(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_sequence_length: int) -> Dict[str, Any]:
    """Tokenize the datasets."""
    return tokenizer(examples["instruction"], padding="max_length", truncation=True, max_length=max_sequence_length)

def create_training_args(config: configparser.ConfigParser) -> TrainingArguments:
    """Create a training arguments object."""
    output_dir = config["DEFAULT"]["output_dir"]
    return TrainingArguments(
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

def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """Define a compute_metrics function."""
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": (predictions == labels).float().mean().item()}

def train_model(model: AutoModelForSequenceClassification, tokenized_train: Any, tokenized_val: Any, training_args: TrainingArguments, compute_metrics: callable) -> None:
    """Train the model."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    trainer.train()

def save_model_and_tokenizer(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, output_dir: str, mini_model_name: str) -> None:
    """Save the model and tokenizer."""
    model_save_path = os.path.join(output_dir, mini_model_name)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(colored(f"Model saved to {model_save_path}", "green"))

def upload_model_to_hugging_face(model_save_path: str, mini_model_name: str) -> None:
    """Upload the model to Hugging Face."""
    logger.info(colored("Uploading the model to Hugging Face...", "green"))
    api = HfApi()
    repo_name = mini_model_name
    repo_url = api.create_repo(repo_name)
    repo = Repository(local_dir=model_save_path, clone_from=repo_url)
    repo.push_to_hub()
    logger.info(colored(f"Model uploaded to {repo_url}", "green"))

def main() -> None:
    """Main function to run the Model Improvement Module."""
    # Load configuration
    config = configparser.ConfigParser()
    try:
        config.read("config.ini")
    except configparser.ParsingError as e:
        logger.error(colored(f"Error parsing config.ini: {e}", "red"))
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

    # Automated loop for suggestions and edits
    while True:
        # Capture terminal output
        terminal_output = capture_terminal_output("python mim.py")

        # Get suggestions from Mistral API
        suggestions = get_suggestions(terminal_output)

        # Apply suggestions to files
        apply_suggestions(suggestions)

        # Wait before re-running the script
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info(colored("Script terminated by user.", "yellow"))
    except Exception as e:
        logger.error(colored(f"An unexpected error occurred: {e}", "red"))
        sys.exit(1)