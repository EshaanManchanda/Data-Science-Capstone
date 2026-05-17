"""
Logging Configuration Module
Sets up logging for the entire project
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_file: str = "logs/project.log", level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_logger(name: str):
    """Get a logger with the specified name"""
    return logging.getLogger(name)


class ProjectLogger:
    """Project-wide logging utility"""

    def __init__(self, name: str = "SpaceX"):
        self.logger = logging.getLogger(name)
        self.name = name

    def info(self, message: str):
        self.logger.info(f"[{self.name}] {message}")

    def warning(self, message: str):
        self.logger.warning(f"[{self.name}] {message}")

    def error(self, message: str):
        self.logger.error(f"[{self.name}] {message}")

    def debug(self, message: str):
        self.logger.debug(f"[{self.name}] {message}")


logger = ProjectLogger("Main")