import pandas as pd

# Verificar train.xlsx
train = pd.read_excel('data/raw/train.xlsx')
print("=== TRAIN.XLSX ===")
print(f"Total registros: {len(train):,}")
if 'ID_CLIENTE' in train.columns:
    print(f"ID_CLIENTE únicos: {train['ID_CLIENTE'].nunique():,}")
    print(f"Múltiplos registros por cliente: {len(train) - train['ID_CLIENTE'].nunique():,}")
else:
    print("ID_CLIENTE não está presente (já foi removido)")

print(f"\nColunas presentes: {train.columns.tolist()[:10]}...")
