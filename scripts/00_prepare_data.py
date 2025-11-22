"""
Script 00: Preparação inicial dos dados crus.
Cria TARGET e divide dados estratificadamente.

Execução:
    conda run -n ml python scripts/00_prepare_data.py
"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_preparation import DataPreparation
from src.utils.logger import logger


def main():
    """Preparar dados crus para pipeline."""
    
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════════════════════════╗")
    logger.info("║                                                                                ║")
    logger.info("║                  📊 ETAPA 00: PREPARAÇÃO DE DADOS CRUS 📊                      ║")
    logger.info("║                                                                                ║")
    logger.info("╚════════════════════════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    # Configurar preparador
    preparer = DataPreparation(
        raw_data_path='data/raw/dataset.xlsx',
        output_dir='data/raw',
        random_state=42  # Seed idêntico ao experimento original
    )
    
    # Executar preparação
    train_path, val_path, test_path = preparer.prepare()
    
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════════════════════════╗")
    logger.info("║                                                                                ║")
    logger.info("║                        ✅ PREPARAÇÃO CONCLUÍDA ✅                              ║")
    logger.info("║                                                                                ║")
    logger.info("║  Próximo passo: python scripts/run_full_pipeline.py                           ║")
    logger.info("║                                                                                ║")
    logger.info("╚════════════════════════════════════════════════════════════════════════════════╝")
    logger.info("")


if __name__ == '__main__':
    main()
