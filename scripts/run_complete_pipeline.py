"""
Pipeline Completo: Do dataset cru até os resultados finais.
Executa todas as etapas:
  00. Preparação de dados (dataset.xlsx → train/val/test splits)
  01. Clustering semântico
  02. Feature engineering  
  03. Treinamento de modelo

Execução:
    conda run -n ml python scripts/run_complete_pipeline.py
"""
import sys
from pathlib import Path
import time

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_preparation import DataPreparation
from src.preprocessing.semantic_clustering import enrich_datasets
from src.preprocessing.feature_engineering import engineer_all_datasets
from src.training.train_gradient_boosting import GradientBoostingTrainer
from src.utils.logger import logger
from src.utils.report_generator import generate_technical_report
from src.config import config


def print_header():
    """Imprime cabeçalho do pipeline."""
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════════════════════════╗")
    logger.info("║                                                                                ║")
    logger.info("║               🚀 PIPELINE COMPLETO - DO ZERO AO RESULTADO 🚀                   ║")
    logger.info("║                                                                                ║")
    logger.info("║  Etapas:                                                                       ║")
    logger.info("║    00. Preparação de dados (dataset.xlsx → splits)                            ║")
    logger.info("║    01. Clustering semântico (Sentence Transformers + KMeans)                  ║")
    logger.info("║    02. Feature engineering (37 features)                                      ║")
    logger.info("║    03. Treinamento Gradient Boosting                                          ║")
    logger.info("║                                                                                ║")
    logger.info("╚════════════════════════════════════════════════════════════════════════════════╝")
    logger.info("")


def step_00_prepare_data() -> tuple:
    """
    ETAPA 00: Preparação de dados crus.
    
    Returns:
        Tuple com caminhos (train_path, validation_path, test_path)
    """
    logger.info("=" * 80)
    logger.info("📊 ETAPA 00/04: PREPARAÇÃO DE DADOS CRUS")
    logger.info("=" * 80)
    
    preparer = DataPreparation(
        raw_data_path='data/raw/dataset.xlsx',
        output_dir='data/raw',
        random_state=42
    )
    
    train_path, val_path, test_path = preparer.prepare()
    
    logger.success("✅ ETAPA 00 CONCLUÍDA")
    logger.info("")
    
    return train_path, val_path, test_path


def step_01_semantic_clustering():
    """ETAPA 01: Clustering semântico."""
    logger.info("=" * 80)
    logger.info("🧠 ETAPA 01/04: CLUSTERING SEMÂNTICO")
    logger.info("=" * 80)
    
    enrich_datasets()
    
    logger.success("✅ ETAPA 01 CONCLUÍDA")
    logger.info("")


def step_02_feature_engineering():
    """ETAPA 02: Feature engineering."""
    logger.info("=" * 80)
    logger.info("⚙️  ETAPA 02/04: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    engineer_all_datasets()
    
    logger.success("✅ ETAPA 02 CONCLUÍDA")
    logger.info("")


def step_03_train_model():
    """ETAPA 03: Treinamento do modelo."""
    logger.info("=" * 80)
    logger.info("🤖 ETAPA 03/04: TREINAMENTO GRADIENT BOOSTING")
    logger.info("=" * 80)
    
    trainer = GradientBoostingTrainer()
    trainer.train()
    
    logger.success("✅ ETAPA 03 CONCLUÍDA")
    logger.info("")


def print_summary(elapsed_time: float):
    """Imprime sumário final."""
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════════════════════════╗")
    logger.info("║                                                                                ║")
    logger.info("║                     🎉 PIPELINE COMPLETO EXECUTADO! 🎉                         ║")
    logger.info("║                                                                                ║")
    logger.info(f"║  ⏱️  Tempo total: {elapsed_time:.1f} minutos                                         ║")
    logger.info("║                                                                                ║")
    logger.info("║  📊 Artefatos gerados:                                                         ║")
    logger.info("║     - data/raw/train.xlsx, validation.xlsx, test.xlsx                         ║")
    logger.info("║     - data/processed/*_with_all_tabs_semantics.xlsx                           ║")
    logger.info("║     - data/features/*_features_engineered.csv                                 ║")
    logger.info("║     - models/experiments/gradient_boosting_churn.joblib                       ║")
    logger.info("║     - outputs/metrics/gb_results.csv                                          ║")
    logger.info("║     - outputs/metrics/gb_feature_importance.csv                               ║")
    logger.info("║                                                                                ║")
    logger.info("║  🔍 Próximo passo:                                                             ║")
    logger.info("║     conda run -n ml python compare_results.py                                 ║")
    logger.info("║                                                                                ║")
    logger.info("╚════════════════════════════════════════════════════════════════════════════════╝")
    logger.info("")


def main():
    """Executar pipeline completo."""
    start_time = time.time()
    
    # Configurar log em arquivo
    from pathlib import Path
    log_dir = Path('outputs/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add_file_handler(log_dir, prefix="complete_pipeline")
    
    print_header()
    
    try:
        # Etapa 00: Preparação de dados
        step_00_prepare_data()
        
        # Etapa 01: Clustering semântico
        step_01_semantic_clustering()
        
        # Etapa 02: Feature engineering
        step_02_feature_engineering()
        
        # Etapa 03: Treinamento do modelo
        step_03_train_model()
        
        # Sumário final
        elapsed_time = (time.time() - start_time) / 60
        print_summary(elapsed_time)
        
        # Gerar relatório técnico
        logger.info("")
        report_path = generate_technical_report(execution_time=elapsed_time)
        logger.info(f"📄 Relatório técnico: {report_path}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ ERRO no pipeline: {str(e)}")
        raise


if __name__ == '__main__':
    main()
