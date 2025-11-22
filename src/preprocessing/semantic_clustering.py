"""
Semantic Clustering com análise de sentimento.
Pattern: Strategy - diferentes estratégias de clustering para TAB_N1, TAB_N2, TAB_N3.
Pattern: Factory - cria estratégias baseadas em configuração.
"""
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import pandas as pd
import torch

from src.config import config
from src.utils.logger import logger
from src.utils.tab_n3_semantics_v2 import TabN3SemanticSentimentModel, export_cluster_dictionary


class ClusteringStrategy(ABC):
    """
    Strategy Pattern: Interface para diferentes estratégias de clustering.
    Cada TAB pode ter configurações distintas (n_clusters, sentiment_seeds).
    """
    
    def __init__(self, tab_name: str):
        self.tab_name = tab_name
        self.model: Optional[TabN3SemanticSentimentModel] = None
        self._config = config.semantic.tab_configs.get(tab_name, {})
        
    @abstractmethod
    def get_n_clusters(self) -> int:
        """Retorna número de clusters para esta estratégia"""
        pass
    
    @abstractmethod
    def get_sentiment_seeds(self) -> Dict[str, float]:
        """Retorna seeds de sentimento para esta estratégia"""
        pass
    
    def fit(self, texts: List[str]) -> "ClusteringStrategy":
        """Treina modelo de clustering + sentiment"""
        print(f"\n{'='*70}")
        print(f"🔬 [{self.tab_name}] Iniciando clustering semântico...")
        print(f"   Textos: {len(texts):,}")
        print(f"   Clusters: {self.get_n_clusters()}")
        print(f"   Device: {config.semantic.device.upper()}")
        print(f"{'='*70}")
        
        logger.progress(f"[{self.tab_name}] Criando modelo (batch_size={config.semantic.batch_size})...")
        self.model = TabN3SemanticSentimentModel(
            n_clusters=self.get_n_clusters(),
            device=config.semantic.device,
            batch_size=config.semantic.batch_size
        )
        
        logger.progress(f"[{self.tab_name}] Gerando embeddings e clustering...")
        self.model.fit(texts, sentiment_seeds=self.get_sentiment_seeds())
        logger.success(f"[{self.tab_name}] Treinamento concluído!")
        return self
    
    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Aplica clustering + sentiment em novos textos"""
        if self.model is None:
            raise ValueError(f"Modelo {self.tab_name} não foi treinado. Execute fit() primeiro.")
        
        logger.progress(f"[{self.tab_name}] Aplicando clustering em {len(texts):,} textos...")
        texts_list = texts.astype(str).fillna("").tolist()
        enriched = self.model.transform(texts_list)
        logger.success(f"[{self.tab_name}] Transform concluído")
        return pd.DataFrame({
            f"{self.tab_name}_GROUP": enriched["semantic_group"],
            f"{self.tab_name}_SENT_SCORE": enriched["sentiment_score"],
            f"{self.tab_name}_SENT_LABEL": enriched["sentiment_label"]
        })
    
    def save(self, path: Path):
        """Salva modelo treinado"""
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"[{self.tab_name}] Modelo salvo: {path}")
    
    def load(self, path: Path) -> "ClusteringStrategy":
        """Carrega modelo treinado"""
        self.model = TabN3SemanticSentimentModel.load(str(path))
        logger.info(f"[{self.tab_name}] Modelo carregado: {path}")
        return self
    
    def export_clusters(self, path: Path):
        """Exporta dicionário de clusters"""
        if self.model is None:
            raise ValueError("Nenhum modelo para exportar")
        path.parent.mkdir(parents=True, exist_ok=True)
        export_cluster_dictionary(self.model, str(path))
        logger.info(f"[{self.tab_name}] Clusters exportados: {path}")


class TabN1Strategy(ClusteringStrategy):
    """Estratégia para TAB_N1: poucos clusters (~6), mais genérico"""
    
    def get_n_clusters(self) -> int:
        return self._config.get("n_clusters", 6)
    
    def get_sentiment_seeds(self) -> Dict[str, float]:
        # Idêntico ao experimento original
        return {
            "cancelamento": -0.8,
            "reclamacao": -0.7,
            "solicitacao": 0.2,
            "agendamento": 0.3,
        }


class TabN2Strategy(ClusteringStrategy):
    """Estratégia para TAB_N2: clusters intermediários (~12)"""
    
    def get_n_clusters(self) -> int:
        return self._config.get("n_clusters", 12)
    
    def get_sentiment_seeds(self) -> Dict[str, float]:
        # Idêntico ao experimento original
        return {
            "inadimplencia": -0.9,
            "cobranca": -0.8,
            "bloqueio": -0.9,
            "problema tecnico": -0.7,
            "instalacao": 0.2,
            "upgrade": 0.4,
            "fidelizacao": 0.5,
        }


class TabN3Strategy(ClusteringStrategy):
    """Estratégia para TAB_N3: clusters detalhados (~18) com granularidade máxima"""
    
    def get_n_clusters(self) -> int:
        return self._config.get("n_clusters", 18)
    
    def get_sentiment_seeds(self) -> Dict[str, float]:
        return {
            "sem sinal": -1.0,
            "sem audio": -1.0,
            "nao navega": -1.0,
            "lentidao": -0.9,
            "oscilando": -0.9,
            "quedas de conexao": -0.9,
            "atendimento ruim": -1.0,
            "revertido por insatisfacao": -1.0,
            "cancelamento": -1.0,
            "desistencia do servico": -0.8,
            "problema tecnico": -0.6,
            "bloqueado por debitos": -0.9,
            "negativacao": -0.9,
            "clube de vantagens": 0.5,
            "oferta de desconto": 0.7,
            "confirmacao de agendamento": 0.2,
            "solicitacao de upgrade": 0.3,
            "nova ordem de instalacao": 0.4,
            "retencao": 0.3,
            "fidelizacao": 0.5,
        }


class ClusteringStrategyFactory:
    """
    Factory Pattern: Cria estratégia apropriada para cada TAB.
    """
    _strategies = {
        "TAB_N1": TabN1Strategy,
        "TAB_N2": TabN2Strategy,
        "TAB_N3": TabN3Strategy,
    }
    
    @classmethod
    def create(cls, tab_name: str) -> ClusteringStrategy:
        """Cria estratégia para o TAB especificado"""
        if tab_name not in cls._strategies:
            raise ValueError(f"TAB desconhecido: {tab_name}. Opções: {list(cls._strategies.keys())}")
        return cls._strategies[tab_name](tab_name)


class SemanticPipeline:
    """
    Pipeline de enriquecimento semântico completo.
    Coordena múltiplas estratégias de clustering.
    """
    
    def __init__(self, tab_names: Optional[List[str]] = None):
        """
        Args:
            tab_names: Lista de TABs para processar. Default: ["TAB_N1", "TAB_N2", "TAB_N3"]
        """
        self.tab_names = tab_names or ["TAB_N1", "TAB_N2", "TAB_N3"]
        self.strategies: Dict[str, ClusteringStrategy] = {
            tab: ClusteringStrategyFactory.create(tab) for tab in self.tab_names
        }
        
        # Log configuração com detalhes de GPU
        print(f"\n{'='*70}")
        print(f"⚙️  CONFIGURAÇÃO DE HARDWARE")
        print(f"{'='*70}")
        print(f"   Device: {config.semantic.device.upper()}")
        if config.semantic.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   CUDA disponível: ✅")
        else:
            print(f"   CUDA disponível: ❌ (usando CPU)")
        print(f"   Batch size: {config.semantic.batch_size}")
        print(f"{'='*70}\n")
    
    def fit(self, df: pd.DataFrame) -> "SemanticPipeline":
        """
        Treina modelos para todos os TABs.
        
        Args:
            df: DataFrame com colunas TAB_N1, TAB_N2, TAB_N3
        """
        logger.info("=== TREINAMENTO DE MODELOS SEMÂNTICOS ===")
        
        for tab_name, strategy in self.strategies.items():
            if tab_name not in df.columns:
                logger.warning(f"Coluna {tab_name} não encontrada. Pulando...")
                continue
            
            texts = df[tab_name].astype(str).fillna("").tolist()
            logger.info(f"[{tab_name}] {len(texts)} textos carregados")
            
            strategy.fit(texts)
            
            # Salvar modelo
            model_path = config.models.experiments / f"{tab_name.lower()}_semantics.joblib"
            strategy.save(model_path)
            
            # Exportar clusters
            cluster_path = config.outputs / "metrics" / f"{tab_name.lower()}_clusters.json"
            strategy.export_clusters(cluster_path)
        
        logger.info("Treinamento concluído para todos os TABs")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica enriquecimento semântico no DataFrame.
        
        Args:
            df: DataFrame com colunas TAB_N1, TAB_N2, TAB_N3
        
        Returns:
            DataFrame enriquecido com colunas *_GROUP, *_SENT_SCORE, *_SENT_LABEL
        """
        df_enriched = df.copy()
        
        for tab_name, strategy in self.strategies.items():
            if tab_name not in df.columns:
                logger.warning(f"Coluna {tab_name} não encontrada. Pulando...")
                continue
            
            enriched_cols = strategy.transform(df[tab_name])
            df_enriched = pd.concat([df_enriched, enriched_cols], axis=1)
            logger.info(f"[{tab_name}] Enriquecimento aplicado")
        
        return df_enriched
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Treina e aplica transformação em um único passo"""
        self.fit(df)
        return self.transform(df)
    
    def save_models(self):
        """Salva todos os modelos treinados"""
        for tab_name, strategy in self.strategies.items():
            model_path = config.models.production / f"{tab_name.lower()}_semantics.joblib"
            strategy.save(model_path)
    
    def load_models(self) -> "SemanticPipeline":
        """Carrega modelos pré-treinados"""
        logger.info("Carregando modelos pré-treinados...")
        for tab_name, strategy in self.strategies.items():
            model_path = config.models.production / f"{tab_name.lower()}_semantics.joblib"
            if model_path.exists():
                strategy.load(model_path)
            else:
                logger.warning(f"Modelo não encontrado: {model_path}")
        return self


def enrich_datasets():
    """
    Função utilitária: Enriquece train/validation/test com clusters semânticos.
    Processa arquivos em data/raw/ e salva em data/processed/.
    """
    logger.info("=== ENRIQUECIMENTO DE DATASETS ===")
    
    # Carregar dados raw
    train_path = config.data.raw / "train.xlsx"
    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {train_path}")
    
    logger.info(f"Carregando dados de treino: {train_path}")
    df_train = pd.read_excel(train_path, engine="openpyxl")
    
    # Treinar pipeline
    pipeline = SemanticPipeline()
    pipeline.fit(df_train)
    
    # Processar todos os splits
    splits = {
        "train": config.data.raw / "train.xlsx",
        "validation": config.data.raw / "validation.xlsx",
        "test": config.data.raw / "test.xlsx",
    }
    
    print(f"\n{'='*70}")
    print(f"🔄 Aplicando clustering nos datasets...")
    print(f"{'='*70}\n")
    
    for i, (split_name, path) in enumerate(splits.items(), 1):
        if not path.exists():
            logger.warning(f"Arquivo não encontrado: {path}. Pulando...")
            continue
        
        print(f"\n[{i}/3] 📂 {split_name.upper()}")
        print(f"   {'─'*66}")
        logger.progress(f"Carregando {path.name}...")
        df = pd.read_excel(path, engine="openpyxl")
        print(f"   Shape: {df.shape}")
        
        df_enriched = pipeline.transform(df)
        
        # Salvar em data/processed/
        output_path = config.data.processed / f"{split_name}_with_all_tabs_semantics.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.progress(f"Salvando em {output_path.name}...")
        df_enriched.to_excel(output_path, index=False)
        logger.success(f"Dataset {split_name} enriquecido!")
    
    print(f"\n{'='*70}")
    print(f"✅ ENRIQUECIMENTO SEMÂNTICO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*70}\n")
