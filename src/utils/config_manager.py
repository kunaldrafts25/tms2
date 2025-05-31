"""
MIT License

Copyright (c) 2024 kunalsingh2514@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
Configuration Manager for Traffic Management System

This module provides centralized configuration management with support for
YAML files, environment variables, and runtime configuration updates.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

class ConfigManager:
    """
    Centralized configuration management system for TMS.
    
    Features:
    - Load configuration from YAML files
    - Environment variable override support
    - Runtime configuration updates
    - Configuration validation
    - Multiple environment support (dev, prod, test)
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration file
            environment: Environment name (development, production, testing)
        """
        self.environment = environment
        self.config_path = config_path or self._get_default_config_path()
        self.config_data: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        base_dir = Path(__file__).parent.parent.parent
        return str(base_dir / "config" / "config.yaml")
    
    def load_config(self) -> None:
        """Load configuration from file and apply environment overrides."""
        try:
            # Load main configuration
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
            
            env_config_path = self._get_env_config_path()
            if os.path.exists(env_config_path):
                with open(env_config_path, 'r', encoding='utf-8') as file:
                    env_config = yaml.safe_load(file)
                    self._merge_configs(self.config_data, env_config)
            
            self._apply_env_overrides()
            
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def _get_env_config_path(self) -> str:
        """Get environment-specific configuration file path."""
        base_dir = Path(self.config_path).parent
        return str(base_dir / f"config.{self.environment}.yaml")
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> None:
        """Recursively merge override configuration into base configuration."""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides using TMS_ prefix."""
        for key, value in os.environ.items():
            if key.startswith('TMS_'):
                config_key = key[4:].lower().replace('_', '.')
                self._set_nested_value(config_key, self._convert_env_value(value))
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number conversion
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def _set_nested_value(self, key_path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = key_path.split('.')
        current = self.config_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get('models.yolo.confidence_threshold', 0.5)
        """
        keys = key_path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        self._set_nested_value(key_path, value)
        self.logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing the section configuration
        """
        return self.get(section, {})
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration (defaults to current config path)
        """
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate configuration for required fields and correct types.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['app', 'models', 'data', 'traffic_signals']
        
        for section in required_sections:
            if section not in self.config_data:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate model paths exist
        yolo_weights = self.get('models.yolo.weights_path')
        if yolo_weights and not os.path.exists(yolo_weights):
            self.logger.warning(f"YOLO weights file not found: {yolo_weights}")
        
        return True
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.load_config()
        self.logger.info("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config_data.copy()
    
    def to_json(self) -> str:
        """Return configuration as JSON string."""
        return json.dumps(self.config_data, indent=2)


# Global configuration instance
_config_instance: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def init_config(config_path: Optional[str] = None, environment: str = "development") -> ConfigManager:
    """Initialize global configuration instance."""
    global _config_instance
    _config_instance = ConfigManager(config_path, environment)
    return _config_instance
