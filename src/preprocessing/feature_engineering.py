"""
Feature Engineering com clusters semânticos.
Pattern: Builder - constrói features incrementalmente através de steps encadeados.
"""
from pathlib import Path
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from src.config import config
from src.utils.logger import logger


class FeatureBuilder(ABC):
    """
    Builder Pattern: Interface para construtores de features.
    Cada builder implementa um tipo específico de feature engineering.
    """
    
    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constrói features e retorna DataFrame com novas colunas"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes das features criadas"""
        pass


class ClusterCounterBuilder(FeatureBuilder):
    """Builder para contadores de clusters críticos"""
    
    def __init__(self):
        self.clusters_criticos = config.feature_engineering.clusters_criticos
        self._feature_names = []
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conta ocorrências de clusters críticos"""
        logger.info("[Builder] Criando contadores de clusters...")
        features = {}
        
        for tab_level, clusters in self.clusters_criticos.items():
            group_col = f"{tab_level}_GROUP"
            
            if group_col not in df.columns:
                logger.warning(f"Coluna {group_col} não encontrada, pulando...")
                continue
            
            for cluster_name, cluster_keywords in clusters.items():
                feature_name = f"count_{tab_level.lower()}_{cluster_name}"
                mask = df[group_col].isin(cluster_keywords)
                features[feature_name] = mask.astype(int)
                self._feature_names.append(feature_name)
                logger.debug(f"{feature_name}: {mask.sum()} ocorrências")
        
        logger.info(f"Criadas {len(features)} features de contadores")
        return pd.DataFrame(features, index=df.index)
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names


class SentimentBuilder(FeatureBuilder):
    """Builder para features de sentimento"""
    
    def __init__(self):
        self._feature_names = []
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de sentimento (proporções e scores)"""
        logger.info("[Builder] Criando features de sentimento...")
        features = {}
        
        for level in [1, 2, 3]:
            tab = f"TAB_N{level}"
            sent_label_col = f"{tab}_SENT_LABEL"
            sent_score_col = f"{tab}_SENT_SCORE"
            
            if sent_label_col not in df.columns:
                continue
            
            # Proporções de sentimento
            for sentiment in ["insatisfacao", "neutro", "positivo"]:
                fname = f"prop_{tab.lower()}_{sentiment}"
                features[fname] = (df[sent_label_col] == sentiment).astype(int)
                self._feature_names.append(fname)
            
            # Score de sentimento
            if sent_score_col in df.columns:
                fname = f"sent_score_{tab.lower()}"
                features[fname] = df[sent_score_col].fillna(0)
                self._feature_names.append(fname)
        
        logger.info(f"Criadas {len(features)} features de sentimento")
        return pd.DataFrame(features, index=df.index)
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names


class EscalationBuilder(FeatureBuilder):
    """Builder para detectar padrões de escalation (piora progressiva)"""
    
    def __init__(self):
        self._feature_names = []
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta escalations entre TABs"""
        logger.info("[Builder] Criando features de escalation...")
        features = {}
        
        # Escalation N1 → N3
        if all(c in df.columns for c in ["TAB_N1_SENT_LABEL", "TAB_N3_SENT_LABEL"]):
            fname = "escalation_n1_to_n3"
            mask = (
                (df["TAB_N1_SENT_LABEL"] == "neutro") & 
                (df["TAB_N3_SENT_LABEL"] == "insatisfacao")
            )
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        # Escalation N2 → N3
        if all(c in df.columns for c in ["TAB_N2_SENT_LABEL", "TAB_N3_SENT_LABEL"]):
            fname = "escalation_n2_to_n3"
            mask = (
                (df["TAB_N2_SENT_LABEL"] == "neutro") & 
                (df["TAB_N3_SENT_LABEL"] == "insatisfacao")
            )
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        # Degradação contínua: positivo/neutro → neutro/insatisfação → insatisfação
        if all(c in df.columns for c in ["TAB_N1_SENT_LABEL", "TAB_N2_SENT_LABEL", "TAB_N3_SENT_LABEL"]):
            fname = "degradacao_continua"
            mask = (
                (df["TAB_N1_SENT_LABEL"].isin(["positivo", "neutro"])) & 
                (df["TAB_N2_SENT_LABEL"].isin(["neutro", "insatisfacao"])) &
                (df["TAB_N3_SENT_LABEL"] == "insatisfacao")
            )
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        logger.info(f"Criadas {len(features)} features de escalation")
        return pd.DataFrame(features, index=df.index)
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names


class CriticalFlagsBuilder(FeatureBuilder):
    """Builder para flags booleanas de situações críticas"""
    
    def __init__(self):
        self._feature_names = []
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria flags de situações críticas"""
        logger.info("[Builder] Criando flags críticas...")
        features = {}
        
        # Flag: problema técnico grave
        if "TAB_N3_GROUP" in df.columns:
            fname = "flag_problema_tecnico_grave"
            mask = df["TAB_N3_GROUP"].isin([
                "sem sinal, sem, sinal, lentidão",
                "não navega, navega, não, adesão",
                "oscilando"
            ])
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        # Flag: teve cobrança/inadimplência
        fname = "flag_teve_cobranca"
        teve_cobranca = pd.Series(False, index=df.index)
        for level in [1, 2, 3]:
            group_col = f"TAB_N{level}_GROUP"
            if group_col in df.columns:
                teve_cobranca |= df[group_col].str.contains("cobrança|inadimplencia", case=False, na=False)
        features[fname] = teve_cobranca.astype(int)
        self._feature_names.append(fname)
        logger.debug(f"{fname}: {teve_cobranca.sum()} casos")
        
        # Flag: mencionou cancelamento
        if "TAB_N3_GROUP" in df.columns:
            fname = "flag_mencionou_cancelamento"
            mask = df["TAB_N3_GROUP"].str.contains("cancelamento", case=False, na=False)
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        # Flag: bloqueado
        if "TAB_N3_GROUP" in df.columns:
            fname = "flag_bloqueado"
            mask = df["TAB_N3_GROUP"].str.contains("bloqueado|desbloqueio", case=False, na=False)
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        # Flag: insatisfação em TAB_N3
        if "TAB_N3_SENT_LABEL" in df.columns:
            fname = "flag_insatisfacao_n3"
            mask = (df["TAB_N3_SENT_LABEL"] == "insatisfacao")
            features[fname] = mask.astype(int)
            self._feature_names.append(fname)
            logger.debug(f"{fname}: {mask.sum()} casos")
        
        logger.info(f"Criadas {len(features)} flags críticas")
        return pd.DataFrame(features, index=df.index)
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names


class AggregatedSentimentBuilder(FeatureBuilder):
    """Builder para features agregadas de sentimento"""
    
    def __init__(self):
        self._feature_names = []
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria agregações de sentiment scores"""
        logger.info("[Builder] Criando agregações de sentimento...")
        features = {}
        
        # Agregações de scores
        score_cols = [f"TAB_N{i}_SENT_SCORE" for i in [1, 2, 3] if f"TAB_N{i}_SENT_SCORE" in df.columns]
        if score_cols:
            for agg_func, suffix in [("mean", "mean"), ("min", "min"), ("max", "max"), ("std", "std")]:
                fname = f"sent_score_{suffix}"
                if agg_func == "std":
                    features[fname] = df[score_cols].std(axis=1).fillna(0)
                else:
                    features[fname] = df[score_cols].agg(agg_func, axis=1)
                self._feature_names.append(fname)
            logger.debug(f"Agregações de scores criadas ({len(score_cols)} TABs)")
        
        # Contagens de sentimentos
        label_cols = [f"TAB_N{i}_SENT_LABEL" for i in [1, 2, 3] if f"TAB_N{i}_SENT_LABEL" in df.columns]
        if label_cols:
            fname_insatis = "count_insatisfacao_total"
            features[fname_insatis] = df[label_cols].apply(
                lambda row: (row == "insatisfacao").sum(), axis=1
            )
            self._feature_names.append(fname_insatis)
            
            fname_posit = "count_positivo_total"
            features[fname_posit] = df[label_cols].apply(
                lambda row: (row == "positivo").sum(), axis=1
            )
            self._feature_names.append(fname_posit)
            logger.debug("Contagens de sentimentos criadas")
        
        logger.info(f"Criadas {len(features)} features agregadas")
        return pd.DataFrame(features, index=df.index)
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names


class FeatureEngineeringDirector:
    """
    Director Pattern: Orquestra execução sequencial dos builders.
    Coordena pipeline completo de feature engineering.
    """
    
    def __init__(self):
        self.builders: List[FeatureBuilder] = [
            ClusterCounterBuilder(),
            SentimentBuilder(),
            EscalationBuilder(),
            CriticalFlagsBuilder(),
            AggregatedSentimentBuilder(),
        ]
    
    def build_features(self, df: pd.DataFrame, preserve_target: bool = True) -> pd.DataFrame:
        """
        Executa pipeline completo de feature engineering.
        
        Args:
            df: DataFrame com clusters semânticos
            preserve_target: Se True, preserva coluna TARGET
        
        Returns:
            DataFrame enriquecido com novas features
        """
        logger.info("=== FEATURE ENGINEERING ===")
        logger.info(f"Shape inicial: {df.shape}")
        
        # Backup TARGET (prevenir corrupção por pd.concat)
        target_backup = None
        if preserve_target and 'TARGET' in df.columns:
            target_backup = df['TARGET'].copy()
            logger.info(f"TARGET preservado: {target_backup.notna().sum()} valores não-NaN")
        
        # Executar builders sequencialmente
        feature_dfs = [df]
        total_features = 0
        
        logger.info(f"Executando {len(self.builders)} builders...")
        for i, builder in enumerate(self.builders, 1):
            try:
                builder_name = builder.__class__.__name__.replace("Builder", "")
                logger.info(f"[{i}/{len(self.builders)}] {builder_name}...")
                features_df = builder.build(df)
                feature_dfs.append(features_df)
                num_features = len(builder.get_feature_names())
                total_features += num_features
                logger.info(f"   ✓ {builder_name}: {num_features} features criadas")
            except Exception as e:
                logger.error(f"Erro em {builder.__class__.__name__}: {e}")
                raise
        
        # Concatenar todas as features
        df_enriched = pd.concat(feature_dfs, axis=1)
        
        # Remover duplicatas de colunas
        df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]
        
        # Restaurar TARGET do backup
        if target_backup is not None:
            df_enriched['TARGET'] = target_backup
            logger.info(f"TARGET restaurado: {df_enriched['TARGET'].notna().sum()} valores não-NaN")
            logger.info(f"Distribuição TARGET: {df_enriched['TARGET'].value_counts().to_dict()}")
        
        logger.info("─" * 70)
        logger.info(f"Total de features criadas: {total_features}")
        logger.info(f"Shape final: {df_enriched.shape}")
        logger.info("=" * 70)
        logger.info("✅ FEATURE ENGINEERING CONCLUÍDO!")
        logger.info("=" * 70)
        
        return df_enriched
    
    def get_all_feature_names(self) -> List[str]:
        """Retorna lista de todas as features criadas por todos os builders"""
        all_features = []
        for builder in self.builders:
            all_features.extend(builder.get_feature_names())
        return all_features


def engineer_features(
    input_path: Path,
    output_path: Path,
    preserve_target: bool = True
) -> pd.DataFrame:
    """
    Aplica feature engineering em um arquivo.
    
    Args:
        input_path: Caminho para arquivo com clusters semânticos
        output_path: Caminho para salvar features
        preserve_target: Se True, preserva TARGET (para train/val/test)
    
    Returns:
        DataFrame com features
    """
    logger.info(f"Processando: {input_path}")
    
    # Carregar dados
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path, engine="openpyxl")
    
    logger.info(f"Dados carregados: {df.shape}")
    
    # Aplicar feature engineering
    director = FeatureEngineeringDirector()
    df_features = director.build_features(df, preserve_target=preserve_target)
    
    # Salvar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".csv":
        df_features.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df_features.to_parquet(output_path, index=False)
    else:
        df_features.to_excel(output_path, index=False)
    
    logger.info(f"Features salvas: {output_path}")
    return df_features


def engineer_all_datasets():
    """
    Função utilitária: Aplica feature engineering em train/validation/test.
    Lê de data/processed/ e salva em data/features/.
    """
    logger.info("=== FEATURE ENGINEERING EM TODOS OS DATASETS ===")
    
    datasets = {
        "train": (
            config.data.processed / "train_with_all_tabs_semantics.xlsx",
            config.data.features / "train_features_engineered.csv"
        ),
        "validation": (
            config.data.processed / "validation_with_all_tabs_semantics.xlsx",
            config.data.features / "validation_features_engineered.csv"
        ),
        "test": (
            config.data.processed / "test_with_all_tabs_semantics.xlsx",
            config.data.features / "test_features_engineered.csv"
        ),
    }
    
    for split_name, (input_path, output_path) in datasets.items():
        if not input_path.exists():
            logger.warning(f"Arquivo não encontrado: {input_path}. Pulando...")
            continue
        
        logger.info(f"\nProcessando {split_name}...")
        engineer_features(input_path, output_path, preserve_target=True)
    
    logger.info("=== FEATURE ENGINEERING CONCLUÍDO EM TODOS OS DATASETS ===")
