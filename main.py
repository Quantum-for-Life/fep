import sys
import argparse
import logging
import config

# from pymbar import MBAR, timeseries


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

    cfg = config.load_config(args.config)

    logging.info(f"Configuration loaded and validated: {cfg}")

    logging.info(f"Temperature value: {cfg.temperature}")
    logging.info(f"Temperature value: {cfg.pressure}")


if __name__ == "__main__":
    sys.exit(main())
