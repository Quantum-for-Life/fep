import sys
import argparse
import tomllib
import logging


def load_config(file_path):
    """Load configuration from a TOML file."""
    with open(file_path, "r") as file:
        config = tomllib.load(file)
    return config


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="CLI App with TOML Config")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logging.info("Configuration loaded:", config)

    logging.info("Temperature value:", config["settings"]["temperature"])
    logging.info("Pressure value:", config["settings"]["pressure"])


if __name__ == "__main__":
    sys.exit(main())
