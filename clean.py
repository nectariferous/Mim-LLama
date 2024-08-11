import json
import logging
import time
import coloredlogs

# Configure logging with colored output
coloredlogs.install(level='DEBUG', fmt='%(asctime)s - %(levelname)s - %(message)s')

def remove_invalid_entries(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        return

    cleaned_data = []

    for entry in data:
        if entry.get('name') == "TestFlight - Apple" and entry.get('logo') == "https://testflight.apple.com/images/testflight-1200_27.jpg":
            logging.info(f"Removing invalid entry: {entry}")
        else:
            cleaned_data.append(entry)

    try:
        with open(file_path, 'w') as file:
            json.dump(cleaned_data, file, indent=4)
        logging.info(f"Cleaned data has been written to {file_path}")
    except IOError as e:
        logging.error(f"Error writing to file {file_path}: {e}")

def main():
    file_path = 'output.json'  # Change this to your actual file path

    while True:
        remove_invalid_entries(file_path)

if __name__ == "__main__":
    main()
