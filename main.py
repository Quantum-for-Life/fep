import sys
import argparse
import tomllib
import logging
from pydantic import BaseModel, ValidationError

# from pymbar import MBAR, timeseries


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


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Calculate FEP")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logging.info(f"Configuration loaded and validated: {config}")

    logging.info(f"Temperature value: {config.temperature}")
    logging.info(f"Temperature value: {config.pressure}")


if __name__ == "__main__":
    sys.exit(main())
