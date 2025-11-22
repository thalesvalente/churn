import pandas as pd

# Carregar dataset cru
df = pd.read_excel('data/raw/dataset.xlsx', engine='openpyxl')

print(f"Shape: {df.shape}")
print(f"\nColunas ({len(df.columns)}):")
print(df.columns.tolist())
print(f"\nPrimeiras 2 linhas:")
print(df.head(2))
print(f"\nInfo sobre ULTIMO_CANCELAMENTO:")
if 'ULTIMO_CANCELAMENTO' in df.columns:
    print(f"  - Não nulos: {df['ULTIMO_CANCELAMENTO'].notna().sum()}")
    print(f"  - Nulos: {df['ULTIMO_CANCELAMENTO'].isna().sum()}")
    print(f"  - Taxa de cancelamento: {df['ULTIMO_CANCELAMENTO'].notna().sum() / len(df) * 100:.2f}%")
else:
    print("  - Coluna não encontrada!")
