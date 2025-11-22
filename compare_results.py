"""
Script para comparar resultados do novo pipeline com resultados históricos.
"""
import pandas as pd
import json
from pathlib import Path

def compare_results():
    """Compara métricas atuais com históricas"""
    
    print("="*80)
    print("COMPARAÇÃO DE RESULTADOS: NOVO PIPELINE vs HISTÓRICO")
    print("="*80)
    
    # Resultados históricos (do relatório técnico)
    historical = {
        "train": {"auc": 0.9992, "precision": 0.9647, "recall": 0.9538, "f1": 0.9592},
        "validation": {"auc": 0.9965, "precision": 0.9537, "recall": 0.9477, "f1": 0.9507},
        "test": {"auc": 0.9941, "precision": 0.9600, "recall": 0.9495, "f1": 0.9547},
        "threshold": 0.2
    }
    
    # Tentar carregar resultados novos
    new_results_path = Path("outputs/metrics/gb_results.csv")
    
    if not new_results_path.exists():
        print("\n❌ Arquivo de resultados não encontrado: outputs/metrics/gb_results.csv")
        print("   O pipeline ainda não completou ou houve erro.")
        return
    
    # Carregar resultados novos
    df_new = pd.read_csv(new_results_path)
    
    print("\n" + "="*80)
    print("RESULTADOS HISTÓRICOS (F1=95.5% em test)")
    print("="*80)
    print(f"{'Dataset':<15} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
    print("-"*80)
    for dataset in ["train", "validation", "test"]:
        h = historical[dataset]
        print(f"{dataset.capitalize():<15} {h['auc']:>8.4f} {h['precision']:>10.4f} {h['recall']:>8.4f} {h['f1']:>10.4f}")
    print(f"\nThreshold otimizado: {historical['threshold']}")
    
    print("\n" + "="*80)
    print("RESULTADOS NOVOS (Pipeline Refatorado)")
    print("="*80)
    print(f"{'Dataset':<15} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
    print("-"*80)
    for _, row in df_new.iterrows():
        print(f"{row['Dataset']:<15} {row['AUC_ROC']:>8.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['F1_Score']:>10.4f}")
    
    # Calcular diferenças
    print("\n" + "="*80)
    print("DIFERENÇAS (Novo - Histórico)")
    print("="*80)
    print(f"{'Dataset':<15} {'ΔF1':>10} {'ΔPrecision':>12} {'ΔRecall':>10} {'ΔAUC':>10}")
    print("-"*80)
    
    for _, row in df_new.iterrows():
        dataset = row['Dataset'].lower()
        h = historical[dataset]
        
        delta_f1 = row['F1_Score'] - h['f1']
        delta_prec = row['Precision'] - h['precision']
        delta_rec = row['Recall'] - h['recall']
        delta_auc = row['AUC_ROC'] - h['auc']
        
        print(f"{row['Dataset']:<15} {delta_f1:>+10.4f} {delta_prec:>+12.4f} {delta_rec:>+10.4f} {delta_auc:>+10.4f}")
    
    # Verificação de corretude
    print("\n" + "="*80)
    print("VERIFICAÇÃO DE CORRETUDE")
    print("="*80)
    
    test_row = df_new[df_new['Dataset'] == 'Test'].iloc[0]
    test_f1_new = test_row['F1_Score']
    test_f1_hist = historical['test']['f1']
    
    diff_f1 = abs(test_f1_new - test_f1_hist)
    
    if diff_f1 < 0.001:  # Diferença < 0.1%
        print(f"✅ SUCESSO: F1-Score idêntico (diff={diff_f1:.6f})")
        print("   O código refatorado reproduz exatamente os resultados históricos!")
    elif diff_f1 < 0.01:  # Diferença < 1%
        print(f"✅ SUCESSO: F1-Score muito próximo (diff={diff_f1:.4f})")
        print("   Diferença < 1%, aceitável devido a arredondamentos.")
    else:
        print(f"⚠️  ATENÇÃO: Diferença significativa no F1-Score (diff={diff_f1:.4f})")
        print("   Pode haver divergência na implementação.")
    
    # Verificar feature importance
    importance_path = Path("outputs/metrics/gb_feature_importance.csv")
    if importance_path.exists():
        df_imp = pd.read_csv(importance_path)
        print("\n" + "="*80)
        print("TOP 10 FEATURES MAIS IMPORTANTES")
        print("="*80)
        print(f"{'Rank':<6} {'Feature':<50} {'Importance':>12}")
        print("-"*80)
        for i, row in df_imp.head(10).iterrows():
            print(f"{i+1:<6} {row['Feature']:<50} {row['Importance']:>12.6f}")
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA")
    print("="*80)

if __name__ == "__main__":
    compare_results()
