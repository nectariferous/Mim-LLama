import os
import subprocess
import sys
import requests
import time
import logging
import configparser
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mistral API credentials and endpoint
MISTRAL_API_KEY = "ZLtWpfzof2FfNJK3AnZjz3yy0ljFR4J7"
MISTRAL_API_URL = "https://api.mistral.ai/v1/suggestions"

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
            logger.info(colored(f"Installing {package}...", "yellow"))
            install_package(package)

# Function to get suggestions from Mistral API
def get_suggestions(terminal_output):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": terminal_output
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("suggestions", [])
    else:
        logger.error(colored(f"Error getting suggestions: {response.text}", "red"))
        return []

# Function to apply suggestions to files
def apply_suggestions(suggestions):
    for suggestion in suggestions:
        file_path = suggestion.get("file_path")
        line_number = suggestion.get("line_number")
        new_content = suggestion.get("new_content")
        if file_path and line_number is not None and new_content:
            with open(file_path, "r") as file:
                lines = file.readlines()
            lines[line_number - 1] = new_content + "\n"
            with open(file_path, "w") as file:
                file.writelines(lines)
            logger.info(colored(f"Applied suggestion to {file_path} at line {line_number}", "green"))

# Function to capture terminal output
def capture_terminal_output(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Main function
def main():
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
    try:
        tokenized_train = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer, max_sequence_length), batched=True, num_proc=os.cpu_count())
        tokenized_val = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer, max_sequence_length), batched=True, num_proc=os.cpu_count())
    except Exception as e:
        logger.error(colored(f"Error during tokenization: {e}", "red"))
        sys.exit(1)

    # Create a training arguments object
    training_args = create_training_args(config)

    # Train the model
    try:
        train_model(model, tokenized_train, tokenized_val, training_args, compute_metrics)
    except Exception as e:
        logger.error(colored(f"Error during training: {e}", "red"))
        sys.exit(1)

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
    main()
