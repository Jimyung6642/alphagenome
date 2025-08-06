#!/usr/bin/env python3
"""
Download Configuration Manager

Manages configuration settings for the AlphaGenome + Tahoe-100M pipeline,
including download timeouts, retry settings, and cache management.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DownloadConfig:
    """
    Configuration manager for download and pipeline settings.
    
    Handles configuration from environment variables, config files,
    and command-line overrides.
    """
    
    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional overrides.
        
        Args:
            overrides: Dictionary of configuration overrides
        """
        self.config = self._load_default_config()
        
        # Load from config.env if available
        self._load_from_env_file()
        
        # Apply command-line overrides
        if overrides:
            self.config.update(overrides)
        
        logger.info("📁 Download configuration initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            # Download settings
            'download_timeout': 3600,  # 1 hour for large datasets
            'api_timeout': 60,
            'retry_attempts': 3,
            'chunk_size': 8192,
            
            # Pipeline limits
            'max_ontologies': None,  # No limit by default
            'max_tfs_per_ontology': None,  # No limit by default  
            'max_tfs': None,  # No limit by default
            'max_cell_lines': None,  # No limit by default
            
            # Cache and output
            'cache_dir': 'comprehensive_cache',
            'output_dir': 'output',
            'tahoe_cache_dir': 'tahoe_cache',
            
            # Analysis settings
            'expression_threshold': 0.1,
            'validate_genes': True,
            'real_data_only': True,
            'comprehensive_tf_discovery': True,
            'strict_data_validation': True,
            
            # Performance settings
            'streaming_mode': True,
            'parallel_processing': True,
            'overnight_mode': False,
            'fast_mode': False,
            'aggressive_caching': False,
            
            # Logging
            'verbose': False,
            'quiet': False
        }
    
    def _load_from_env_file(self):
        """Load configuration from config.env file if it exists."""
        config_file = Path("config.env")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Handle specific environment variables
                            if key in ['DOWNLOAD_TIMEOUT', 'API_TIMEOUT', 'RETRY_ATTEMPTS']:
                                try:
                                    self.config[key.lower()] = int(value)
                                except ValueError:
                                    logger.warning(f"Invalid integer value for {key}: {value}")
                            elif key in ['EXPRESSION_THRESHOLD']:
                                try:
                                    self.config[key.lower()] = float(value)
                                except ValueError:
                                    logger.warning(f"Invalid float value for {key}: {value}")
                            elif key in ['VERBOSE', 'QUIET', 'FAST_MODE', 'OVERNIGHT_MODE']:
                                self.config[key.lower()] = value.lower() in ['true', '1', 'yes']
                
                logger.info("✅ Loaded configuration from config.env")
            except Exception as e:
                logger.warning(f"Failed to load config.env: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()
    
    def validate_configuration(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate timeout values
        if self.config['download_timeout'] <= 0:
            errors.append("download_timeout must be positive")
        
        if self.config['api_timeout'] <= 0:
            errors.append("api_timeout must be positive")
        
        if self.config['retry_attempts'] < 0:
            errors.append("retry_attempts must be non-negative")
        
        # Validate threshold values
        if self.config['expression_threshold'] < 0:
            errors.append("expression_threshold must be non-negative")
        
        # Validate directory paths
        required_dirs = ['cache_dir', 'output_dir', 'tahoe_cache_dir']
        for dir_key in required_dirs:
            if not self.config[dir_key]:
                errors.append(f"{dir_key} cannot be empty")
        
        if errors:
            logger.error(f"Configuration validation failed: {'; '.join(errors)}")
            return False
        
        logger.info("✅ Configuration validation passed")
        return True
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.config['cache_dir'],
            self.config['output_dir'], 
            self.config['tahoe_cache_dir']
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(exist_ok=True)
                logger.info(f"📁 Ensured directory exists: {dir_path}")
    
    def get_summary(self) -> str:
        """Get a summary of current configuration."""
        summary_lines = [
            "📊 Download Configuration Summary:",
            f"   Download timeout: {self.config['download_timeout']/3600:.1f} hours",
            f"   API timeout: {self.config['api_timeout']} seconds",
            f"   Retry attempts: {self.config['retry_attempts']}",
            f"   Expression threshold: {self.config['expression_threshold']}",
            f"   Max ontologies: {self.config['max_ontologies'] or 'ALL'}",
            f"   Max TFs per ontology: {self.config['max_tfs_per_ontology'] or 'ALL'}",
            f"   Cache directory: {self.config['cache_dir']}",
            f"   Output directory: {self.config['output_dir']}",
            f"   Tahoe cache: {self.config['tahoe_cache_dir']}",
            f"   Fast mode: {self.config['fast_mode']}",
            f"   Overnight mode: {self.config['overnight_mode']}"
        ]
        return '\n'.join(summary_lines)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"DownloadConfig({len(self.config)} settings)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"DownloadConfig(config={self.config})"


# Convenience function for quick access
def create_config(overrides: Optional[Dict[str, Any]] = None) -> DownloadConfig:
    """
    Create a DownloadConfig instance with optional overrides.
    
    Args:
        overrides: Configuration overrides
        
    Returns:
        DownloadConfig instance
    """
    return DownloadConfig(overrides)


if __name__ == "__main__":
    # Test the configuration
    print("🧪 Testing DownloadConfig...")
    
    config = DownloadConfig()
    print(config.get_summary())
    
    # Test validation
    is_valid = config.validate_configuration()
    print(f"Configuration valid: {is_valid}")
    
    # Test directory creation
    config.create_directories()
    
    print("✅ DownloadConfig test completed!")