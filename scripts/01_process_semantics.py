"""
Script 01: Processar semântica (clustering + sentiment)

Executa:
- Carrega train.xlsx de data/raw/
- Treina modelos semânticos para TAB_N1, TAB_N2, TAB_N3
- Enriquece train/validation/test com clusters e sentimentos
- Salva em data/processed/ arquivos *_with_all_tabs_semantics.xlsx
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.semantic_clustering import enrich_datasets
from src.utils.logger import logger
from src.config import config


def main():
    """Pipeline de enriquecimento semântico"""
    config.ensure_paths_exist()
    
    log_dir = config.outputs / "logs"
    logger.add_file_handler(log_dir, prefix="01_semantics")
    
    logger.info("="*80)
    logger.info("SCRIPT 01: PROCESSAMENTO SEMÂNTICO")
    logger.info("="*80)
    
    try:
        enrich_datasets()
        logger.info("\n✓ Pipeline semântico concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro no processamento semântico: {e}")
        raise


if __name__ == "__main__":
    main()
