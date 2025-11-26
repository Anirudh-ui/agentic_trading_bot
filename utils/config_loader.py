import yaml
import os # <-- Must import os

# Define the base directory (where this script, config_loader.py, lives)
# This will be /app/utils/ inside the Docker container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the config path relative to the script
# This correctly resolves to /app/config/config.yaml
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, '..', 'config', 'config.yaml')

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    # Use os.path.normpath to ensure the path is correctly formatted for the OS (Linux in Docker)
    normalized_path = os.path.normpath(config_path) 
    
    with open(normalized_path, "r") as file:
        config = yaml.safe_load(file)
    return config