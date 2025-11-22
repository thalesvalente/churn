"""
Script 03: Treinar Modelo

Executa:
- Carrega *_features_engineered.csv de data/features/
- Treina Gradient Boosting Classifier
- Otimiza threshold no validation
- Avalia em train/val/test
- Salva modelo em models/experiments/
- Salva métricas em outputs/metrics/
- Para produção: copiar manualmente para models/production/
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_gradient_boosting import train_gradient_boosting
from src.utils.logger import logger
from src.config import config


def main():
    """Pipeline de treinamento"""
    config.ensure_paths_exist()
    
    log_dir = config.outputs / "logs"
    logger.add_file_handler(log_dir, prefix="03_train")
    
    logger.info("="*80)
    logger.info("SCRIPT 03: TREINAMENTO DE MODELO")
    logger.info("="*80)
    
    try:
        results = train_gradient_boosting()
        
        logger.info("\n✓ Treinamento concluído com sucesso!")
        logger.info(f"Threshold otimizado: {results['threshold']}")
        logger.info(f"F1-Score (test): {results['test']['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        raise


if __name__ == "__main__":
    main()
