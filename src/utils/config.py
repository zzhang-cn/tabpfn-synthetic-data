"""Configuration management for synthetic data generation."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for synthetic data generation.
    
    This class handles loading and accessing configuration parameters
    from YAML files with support for nested access and updates.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        
        # Set up logging based on config
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get(
            'format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logging.basicConfig(level=level, format=format_str)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key (e.g., 'dataset.n_samples.max')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Set config {key} = {value}")
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates (can be nested)
        """
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = recursive_update(self.config, updates)
        logger.debug(f"Updated configuration with {len(updates)} top-level keys")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def to_json(self, indent: int = 2) -> str:
        """Return configuration as JSON string."""
        return json.dumps(self.config, indent=indent)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {path}")
    
    def __repr__(self) -> str:
        return f"Config(keys={list(self.config.keys())})"
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False)
