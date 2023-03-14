import yaml

def load_config(filename: str) -> dict:
    """ Load a YAML config file 
    
    Args:
        filename (str): Name of file to open
    
    Returns:
        config (dict): Loaded configuration dictionary
    """
    with open(filename) as fh:
        config = yaml.load(fh, yaml.Loader)
    return config
