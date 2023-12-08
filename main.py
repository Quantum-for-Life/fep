import argparse
import tomllib

def load_config(file_path):
    """Load configuration from a TOML file."""
    with open(file_path, 'r') as file:
        config = tomllib.load(file)
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CLI App with TOML Config')
    parser.add_argument('-c', '--config', type=str, default='config.toml',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("Configuration loaded:", config)
    
    print("Temperature value:", config['settings']['temperature'])
    print("Pressure value:", config['settings']['pressure'])

if __name__ == '__main__':
    main()
