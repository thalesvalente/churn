import pandas as pd

df = pd.read_excel('data/raw/dataset.xlsx')

print(f'Total registros: {len(df):,}')
print(f'ID_CLIENTE únicos: {df["ID_CLIENTE"].nunique():,}')
print(f'Registros duplicados: {len(df) - df["ID_CLIENTE"].nunique():,}')
print(f'Média atendimentos/cliente: {len(df) / df["ID_CLIENTE"].nunique():.2f}')
print(f'\nAlguns clientes com múltiplos registros:')
print(df.groupby('ID_CLIENTE').size().sort_values(ascending=False).head(10))
