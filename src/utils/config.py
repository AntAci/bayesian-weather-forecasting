"""
Configuration utilities for loading and managing YAML configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_config_with_base(config_path: str) -> Dict[str, Any]:
    """
    Load configuration with base config inheritance.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Merged configuration dictionary
    """
    # Load base config
    base_path = Path(config_path).parent / "base.yaml"
    if base_path.exists():
        base_config = load_config(base_path)
    else:
        base_config = {}
    
    # Load specific config
    specific_config = load_config(config_path)
    
    # Merge configs (specific overrides base)
    merged_config = {**base_config, **specific_config}
    
    return merged_config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for key, value in config.items():
        if isinstance(value, list):
            print(f"{key}: {value}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print()

if __name__ == "__main__":
    # Test the config loader
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        try:
            config = load_config_with_base(config_path)
            print_config(config, f"Loaded: {config_path}")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print("Usage: python config.py <config_path>")
        print("Example: python config.py configs/bnn_mc.yaml")
