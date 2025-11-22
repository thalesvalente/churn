"""
Logger centralizado do projeto.
Pattern: Singleton + Observer (pode adicionar handlers dinamicamente).
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """Logger singleton com suporte a arquivo e console"""
    _instance: Optional["Logger"] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is not None:
            return
        
        self._logger = logging.getLogger("em_churn")
        self._logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
    
    def add_file_handler(self, log_dir: Path, prefix: str = "pipeline"):
        """Adiciona handler de arquivo"""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{prefix}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        
        self.info(f"Log file: {log_file}")
    
    def info(self, msg: str):
        self._logger.info(msg)
    
    def debug(self, msg: str):
        self._logger.debug(msg)
    
    def warning(self, msg: str):
        self._logger.warning(msg)
    
    def error(self, msg: str):
        self._logger.error(msg)
    
    def critical(self, msg: str):
        self._logger.critical(msg)
    
    def progress(self, msg: str):
        """Log de progresso com indicador visual"""
        self._logger.info(msg)
    
    def success(self, msg: str):
        """Log de sucesso com indicador visual"""
        self._logger.info(msg)


# Instância global
logger = Logger()
