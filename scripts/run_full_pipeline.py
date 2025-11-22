"""
Script FULL PIPELINE: Executa pipeline completo

Executa sequencialmente:
1. Processamento semântico (01_process_semantics.py)
2. Feature engineering (02_engineer_features.py)
3. Treinamento de modelo (03_train_model.py)

Resultado final:
- Modelo treinado em models/experiments/gradient_boosting_churn.joblib
- Métricas em outputs/metrics/gb_results.csv
- Feature importance em outputs/metrics/gb_feature_importance.csv
"""
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.semantic_clustering import enrich_datasets
from src.preprocessing.feature_engineering import engineer_all_datasets
from src.training.train_gradient_boosting import train_gradient_boosting
from src.utils.logger import logger
from src.config import config


def main():
    """Executa pipeline completo"""
    config.ensure_paths_exist()
    
    log_dir = config.outputs / "logs"
    logger.add_file_handler(log_dir, prefix="full_pipeline")
    
    start_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"{'🚀 FULL PIPELINE: CHURN PREDICTION 🚀':^80}")
    print(f"{'='*80}")
    print(f"Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # Step 1: Semantic Clustering
        print(f"\n{'█'*80}")
        print(f"█ STEP 1/3: PROCESSAMENTO SEMÂNTICO (Clustering + Sentiment) {'█':>13}")
        print(f"{'█'*80}\n")
        enrich_datasets()
        
        # Step 2: Feature Engineering
        print(f"\n{'█'*80}")
        print(f"█ STEP 2/3: FEATURE ENGINEERING (37 Features) {'█':>26}")
        print(f"{'█'*80}\n")
        engineer_all_datasets()
        
        # Step 3: Model Training
        print(f"\n{'█'*80}")
        print(f"█ STEP 3/3: TREINAMENTO DO MODELO (Gradient Boosting) {'█':>15}")
        print(f"{'█'*80}\n")
        results = train_gradient_boosting()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"{'🎉 PIPELINE CONCLUÍDO COM SUCESSO! 🎉':^80}")
        print(f"{'='*80}\n")
        print(f"⏱️  Duração total: {duration:.1f}s ({duration/60:.1f} min)\n")
        print(f"📊 Resultados finais (TEST SET):\n")
        print(f"   {'Métrica':<20} {'Valor':>10}")
        print(f"   {'-'*32}")
        print(f"   {'F1-Score':<20} {results['test']['f1']:>10.4f} ⭐")
        print(f"   {'AUC-ROC':<20} {results['test']['auc']:>10.4f}")
        print(f"   {'Precision':<20} {results['test']['precision']:>10.4f}")
        print(f"   {'Recall':<20} {results['test']['recall']:>10.4f}")
        print(f"   {'Threshold':<20} {results['threshold']:>10.1f}")
        print(f"\n💾 Modelo salvo: models/experiments/gradient_boosting_churn.joblib")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
