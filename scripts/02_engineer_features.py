"""
Script 02: Feature Engineering

Executa:
- Carrega *_with_all_tabs_semantics.xlsx de data/processed/
- Aplica feature engineering (37 features)
- Salva em data/features/ arquivos *_features_engineered.csv
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import engineer_all_datasets
from src.utils.logger import logger
from src.config import config


def main():
    """Pipeline de feature engineering"""
    config.ensure_paths_exist()
    
    log_dir = config.outputs / "logs"
    logger.add_file_handler(log_dir, prefix="02_features")
    
    logger.info("="*80)
    logger.info("SCRIPT 02: FEATURE ENGINEERING")
    logger.info("="*80)
    
    try:
        engineer_all_datasets()
        logger.info("\n✓ Feature engineering concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro no feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()
