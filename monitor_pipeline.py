"""
Monitor de progresso do pipeline completo.
Exibe status em tempo real das 4 etapas.
"""
import sys
from pathlib import Path
from datetime import datetime
import time

def get_latest_log():
    """Retorna o log mais recente."""
    log_dir = Path('outputs/logs')
    logs = list(log_dir.glob('complete_pipeline_*.log'))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)

def parse_log_status(log_path):
    """Extrai status das etapas do log."""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    etapas = {
        0: "⏳ Aguardando",
        1: "⏳ Aguardando", 
        2: "⏳ Aguardando",
        3: "⏳ Aguardando"
    }
    
    # ETAPA 00
    if "ETAPA 00/04" in content:
        etapas[0] = "🔄 Em progresso"
    if "✅ ETAPA 00 CONCLUÍDA" in content:
        etapas[0] = "✅ Concluída"
    
    # ETAPA 01
    if "ETAPA 01/04" in content:
        etapas[1] = "🔄 Em progresso"
    if "✅ ETAPA 01 CONCLUÍDA" in content:
        etapas[1] = "✅ Concluída"
    
    # ETAPA 02
    if "ETAPA 02/04" in content:
        etapas[2] = "🔄 Em progresso"
    if "✅ ETAPA 02 CONCLUÍDA" in content:
        etapas[2] = "✅ Concluída"
    
    # ETAPA 03
    if "ETAPA 03/04" in content:
        etapas[3] = "🔄 Em progresso"
    if "✅ ETAPA 03 CONCLUÍDA" in content or "TREINAMENTO CONCLUÍDO" in content:
        etapas[3] = "✅ Concluída"
    
    # Pipeline completo
    pipeline_completo = "🎉 PIPELINE COMPLETO EXECUTADO!" in content
    
    return etapas, pipeline_completo

def check_artifacts():
    """Verifica artefatos gerados."""
    artifacts = {
        "train.xlsx": Path("data/raw/train.xlsx").exists(),
        "validation.xlsx": Path("data/raw/validation.xlsx").exists(),
        "test.xlsx": Path("data/raw/test.xlsx").exists(),
        "train_processed": Path("data/processed/train_with_all_tabs_semantics.xlsx").exists(),
        "train_features": Path("data/features/train_features_engineered.csv").exists(),
        "gb_model": Path("models/production/gradient_boosting_churn.joblib").exists(),
        "gb_results": Path("outputs/metrics/gb_results.csv").exists(),
    }
    return artifacts

def display_status():
    """Exibe status do pipeline."""
    log_path = get_latest_log()
    
    if not log_path:
        print("❌ Nenhum log encontrado!")
        return False
    
    print("\n" + "="*80)
    print(f"📊 MONITOR DO PIPELINE - {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    print(f"\n📝 Log: {log_path.name}")
    print(f"🕒 Modificado: {datetime.fromtimestamp(log_path.stat().st_mtime).strftime('%H:%M:%S')}")
    print(f"📦 Tamanho: {log_path.stat().st_size / 1024:.2f} KB")
    
    etapas, completo = parse_log_status(log_path)
    
    print("\n" + "─"*80)
    print("📋 STATUS DAS ETAPAS:")
    print("─"*80)
    print(f"  00. Preparação de dados        {etapas[0]}")
    print(f"  01. Clustering semântico       {etapas[1]}")
    print(f"  02. Feature engineering        {etapas[2]}")
    print(f"  03. Treinamento GB             {etapas[3]}")
    
    artifacts = check_artifacts()
    
    print("\n" + "─"*80)
    print("📁 ARTEFATOS GERADOS:")
    print("─"*80)
    print(f"  data/raw/splits               {'✅' if artifacts['train.xlsx'] else '⏳'}")
    print(f"  data/processed/*_semantics    {'✅' if artifacts['train_processed'] else '⏳'}")
    print(f"  data/features/*_engineered    {'✅' if artifacts['train_features'] else '⏳'}")
    print(f"  models/production/gb_model    {'✅' if artifacts['gb_model'] else '⏳'}")
    print(f"  outputs/metrics/gb_results    {'✅' if artifacts['gb_results'] else '⏳'}")
    
    if completo:
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
        print("="*80)
        
        if artifacts['gb_results']:
            print("\n📊 Resultados:")
            results_path = Path("outputs/metrics/gb_results.csv")
            with open(results_path, 'r') as f:
                print(f.read())
        
        return True
    
    print("\n" + "="*80)
    return False

def main():
    """Loop de monitoramento."""
    import sys
    
    # Verificar se é execução única ou monitoramento contínuo
    continuous = "--watch" in sys.argv or "-w" in sys.argv
    
    if continuous:
        print("🚀 Iniciando monitoramento contínuo...")
        print("   Ctrl+C para interromper\n")
    
    try:
        while True:
            completo = display_status()
            if completo or not continuous:
                break
            time.sleep(10)  # Atualiza a cada 10 segundos
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoramento interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")

if __name__ == '__main__':
    main()
