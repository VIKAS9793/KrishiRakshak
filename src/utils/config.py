"""
Configuration loader and validator for Krishi Rakshak project.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class ProjectConfig:
    """Class to load and validate project configuration."""
    
    REQUIRED_FIELDS = [
        "project_name",
        "tagline",
        "description",
        "deadline",
        "components",
        "dataset",
        "language_support",
        "license"
    ]
    
    def __init__(self, config_path: str = "project_config.json"):
        """Initialize configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file.
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load and validate the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in self.config]
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {', '.join(missing_fields)}")
        
        # Validate components
        if not isinstance(self.config.get("components"), list):
            raise ValueError("Components must be a list")
        
        # Validate dataset
        dataset = self.config.get("dataset", {})
        if not all(key in dataset for key in ["name", "source", "type"]):
            raise ValueError("Dataset configuration is incomplete")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)
    
    def get_component(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a component configuration by name."""
        for component in self.config.get("components", []):
            if component.get("name") == name:
                return component
        return None
    
    @property
    def project_info(self) -> Dict[str, Any]:
        """Get basic project information."""
        return {
            "name": self.get("project_name"),
            "tagline": self.get("tagline"),
            "description": self.get("description"),
            "deadline": self.get("deadline"),
            "license": self.get("license")
        }
    
    @property
    def dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return self.get("dataset", {})
    
    @property
    def components_info(self) -> List[Dict[str, Any]]:
        """Get list of all components."""
        return self.get("components", [])

# Singleton instance
_config_instance = None

def get_config(config_path: str = "project_config.json") -> ProjectConfig:
    """Get or create a singleton instance of ProjectConfig."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ProjectConfig(config_path)
    return _config_instance
