"""
Módulo de preparação de dados crus.
Responsável por criar TARGET e dividir dados estratificadamente.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
from src.utils.logger import logger


class DataPreparation:
    """
    Prepara dados crus para o pipeline de ML.
    
    Responsabilidades:
    - Criar coluna TARGET binária a partir de ULTIMO_CANCELAMENTO
    - Remover colunas desnecessárias para modelagem
    - Dividir dados estratificadamente (80% train, 10% val, 10% test)
    """
    
    COLUMNS_TO_DROP = [
        'ULTIMO_CANCELAMENTO',  # Usado para criar TARGET, depois removido
        'ID_CLIENTE',           # Identificador, não feature
        'CODIGO',               # Identificador interno
        'PROTOCOLO',            # Identificador de atendimento
        'NUMERO_OS',            # Identificador de ordem de serviço
        'SITUACAO'              # Data leakage - indica status atual do cliente
    ]
    
    def __init__(self, raw_data_path: str, output_dir: str, random_state: int = 42):
        """
        Args:
            raw_data_path: Caminho para dataset.xlsx cru
            output_dir: Diretório para salvar train/validation/test splits
            random_state: Seed para reprodutibilidade
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Criar diretório de saída se não existir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 DataPreparation inicializado")
        logger.info(f"   Dataset cru: {self.raw_data_path}")
        logger.info(f"   Saída: {self.output_dir}")
        logger.info(f"   Random state: {self.random_state}")
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria coluna TARGET binária.
        TARGET = 1 se ULTIMO_CANCELAMENTO não é nulo (cliente cancelou)
        TARGET = 0 se ULTIMO_CANCELAMENTO é nulo (cliente ativo)
        
        Args:
            df: DataFrame com coluna ULTIMO_CANCELAMENTO
            
        Returns:
            DataFrame com coluna TARGET adicionada
        """
        logger.progress("🎯 Criando coluna TARGET...")
        
        # Criar TARGET: 1 se cancelou, 0 se não cancelou
        df['TARGET'] = df['ULTIMO_CANCELAMENTO'].notna().astype(int)
        
        # Estatísticas
        n_total = len(df)
        n_churn = df['TARGET'].sum()
        n_active = n_total - n_churn
        pct_churn = (n_churn / n_total) * 100
        
        logger.info(f"   Total: {n_total:,} registros")
        logger.info(f"   Ativos (TARGET=0): {n_active:,} ({100-pct_churn:.2f}%)")
        logger.info(f"   Cancelados (TARGET=1): {n_churn:,} ({pct_churn:.2f}%)")
        
        return df
    
    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas que não serão usadas na modelagem.
        
        Args:
            df: DataFrame completo
            
        Returns:
            DataFrame sem colunas desnecessárias
        """
        logger.progress("🗑️  Removendo colunas desnecessárias...")
        
        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        
        logger.info(f"   Removendo: {cols_to_drop}")
        df_clean = df.drop(columns=cols_to_drop)
        
        logger.info(f"   Shape: {df.shape} → {df_clean.shape}")
        
        return df_clean
    
    def stratified_split(
        self, 
        df: pd.DataFrame,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide dados estratificadamente mantendo proporção de TARGET.
        
        Args:
            df: DataFrame com coluna TARGET
            train_size: Proporção para treino (default: 0.8)
            val_size: Proporção para validação (default: 0.1)
            test_size: Proporção para teste (default: 0.1)
            
        Returns:
            Tuple com (train_df, validation_df, test_df)
        """
        logger.progress("📊 Divisão estratificada dos dados...")
        
        # Validar proporções
        assert abs((train_size + val_size + test_size) - 1.0) < 0.001, \
            "Soma das proporções deve ser 1.0"
        
        # Primeira divisão: train vs (validation + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            stratify=df['TARGET'],
            random_state=self.random_state
        )
        
        # Segunda divisão: validation vs test
        # Proporção relativa: test_size / (val_size + test_size)
        relative_test_size = test_size / (val_size + test_size)
        
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            stratify=temp_df['TARGET'],
            random_state=self.random_state
        )
        
        # Log estatísticas
        logger.info(f"   Train: {len(train_df):,} registros ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"      └─ TARGET=0: {(train_df['TARGET']==0).sum():,}")
        logger.info(f"      └─ TARGET=1: {(train_df['TARGET']==1).sum():,}")
        
        logger.info(f"   Validation: {len(validation_df):,} registros ({len(validation_df)/len(df)*100:.1f}%)")
        logger.info(f"      └─ TARGET=0: {(validation_df['TARGET']==0).sum():,}")
        logger.info(f"      └─ TARGET=1: {(validation_df['TARGET']==1).sum():,}")
        
        logger.info(f"   Test: {len(test_df):,} registros ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"      └─ TARGET=0: {(test_df['TARGET']==0).sum():,}")
        logger.info(f"      └─ TARGET=1: {(test_df['TARGET']==1).sum():,}")
        
        return train_df, validation_df, test_df
    
    def prepare(self) -> Tuple[Path, Path, Path]:
        """
        Pipeline completo de preparação de dados.
        
        Returns:
            Tuple com caminhos para (train.xlsx, validation.xlsx, test.xlsx)
        """
        logger.info("=" * 80)
        logger.info("🔧 PREPARAÇÃO DE DADOS CRUS")
        logger.info("=" * 80)
        
        # 1. Carregar dados crus
        logger.progress(f"📥 Carregando dataset cru: {self.raw_data_path}")
        df = pd.read_excel(self.raw_data_path, engine='openpyxl')
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Colunas: {len(df.columns)}")
        
        # 2. Criar TARGET
        df = self.create_target(df)
        
        # 3. Remover colunas desnecessárias
        df = self.remove_unnecessary_columns(df)
        
        # 4. Dividir estratificadamente
        train_df, validation_df, test_df = self.stratified_split(df)
        
        # 5. Salvar splits
        logger.progress("💾 Salvando splits...")
        
        train_path = self.output_dir / 'train.xlsx'
        validation_path = self.output_dir / 'validation.xlsx'
        test_path = self.output_dir / 'test.xlsx'
        
        train_df.to_excel(train_path, index=False)
        logger.info(f"   ✅ Train: {train_path}")
        
        validation_df.to_excel(validation_path, index=False)
        logger.info(f"   ✅ Validation: {validation_path}")
        
        test_df.to_excel(test_path, index=False)
        logger.info(f"   ✅ Test: {test_path}")
        
        logger.success("✅ PREPARAÇÃO DE DADOS CONCLUÍDA")
        logger.info("=" * 80)
        
        return train_path, validation_path, test_path


def main():
    """Executar preparação de dados standalone."""
    from src.config import Config
    
    config = Config.get_instance()
    
    preparer = DataPreparation(
        raw_data_path='data/raw/dataset.xlsx',
        output_dir='data/raw',
        random_state=42
    )
    
    train_path, val_path, test_path = preparer.prepare()
    
    print(f"\n✅ Splits criados:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"   - {test_path}")


if __name__ == '__main__':
    main()
