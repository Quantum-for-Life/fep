import tomllib
import logging
from pydantic import BaseModel, ValidationError

# Define a Pydantic model for configuration
class Settings(BaseModel):
    temperature: float
    pressure: float


def load_config(file_path):
    """Load and validate configuration from a TOML file using Pydantic."""
    try:
        with open(file_path, "rb") as file:
            config_data = tomllib.load(file)
        return Settings(**config_data["settings"])
    except ValidationError as e:
        logging.error("Configuration validation error: %s", e)
        raise
