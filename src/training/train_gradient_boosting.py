"""
Treinamento de Gradient Boosting para predição de churn.
Pattern: Template Method - define skeleton do treinamento, subclasses customizam steps.
"""
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from src.config import config
from src.utils.logger import logger


class GradientBoostingTrainer:
    """
    Template Method Pattern: Define estrutura do treinamento.
    Permite customização de steps específicos através de herança.
    """
    
    def __init__(
        self,
        n_estimators: int = None,
        learning_rate: float = None,
        max_depth: int = None,
        random_state: int = None,
        optimize_threshold: bool = None,
        threshold_range: list = None
    ):
        """
        Args:
            n_estimators: Número de árvores (default: config)
            learning_rate: Taxa de aprendizado (default: config)
            max_depth: Profundidade máxima (default: config)
            random_state: Seed aleatória (default: config)
            optimize_threshold: Se True, otimiza threshold no validation (default: config)
            threshold_range: Lista de thresholds para testar (default: config)
        """
        cfg = config.training
        self.n_estimators = n_estimators or cfg.n_estimators
        self.learning_rate = learning_rate or cfg.learning_rate
        self.max_depth = max_depth or cfg.max_depth
        self.random_state = random_state or cfg.random_state
        self.optimize_threshold = optimize_threshold if optimize_threshold is not None else cfg.threshold_optimization
        self.threshold_range = threshold_range or cfg.threshold_range
        
        self.model: Optional[GradientBoostingClassifier] = None
        self.best_threshold: float = 0.5
        self.feature_names: list = []
        
        # Path padrão para salvar modelo (experiments)
        self.model_save_path = config.models.experiments / "gradient_boosting_churn.joblib"
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Template step: Carrega datasets"""
        logger.info("Carregando datasets...")
        
        train_path = config.data.features / "train_features_engineered.csv"
        val_path = config.data.features / "validation_features_engineered.csv"
        test_path = config.data.features / "test_features_engineered.csv"
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        
        logger.info(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
        return df_train, df_val, df_test
    
    def prepare_features(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Tuple:
        """Template step: Prepara features numéricas"""
        logger.info("Preparando features...")
        
        # Extrair TARGET
        y_train = df_train["TARGET"].fillna(0).astype(int)
        y_val = df_val["TARGET"].fillna(0).astype(int)
        y_test = df_test["TARGET"].fillna(0).astype(int)
        
        # Remover colunas não-numéricas
        X_train = df_train.drop(columns=config.columns_to_remove, errors='ignore')
        X_val = df_val.drop(columns=config.columns_to_remove, errors='ignore')
        X_test = df_test.drop(columns=config.columns_to_remove, errors='ignore')
        
        # Apenas features numéricas
        X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
        X_val = X_val.select_dtypes(include=[np.number]).fillna(0)
        X_test = X_test.select_dtypes(include=[np.number]).fillna(0)
        
        self.feature_names = X_train.columns.tolist()
        
        logger.info(f"Features numéricas: {X_train.shape[1]}")
        logger.info(f"Distribuição TARGET (train): {y_train.value_counts().to_dict()}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_model(self) -> GradientBoostingClassifier:
        """Template step: Constrói modelo"""
        logger.info("Construindo Gradient Boosting Classifier...")
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=0
        )
    
    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Template step: Treina modelo"""
        logger.info("Treinando modelo...")
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        logger.info("Treinamento concluído")
    
    def optimize_threshold_on_validation(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Template step: Otimiza threshold no validation"""
        if not self.optimize_threshold:
            logger.info("Otimização de threshold desabilitada. Usando 0.5.")
            return
        
        logger.info("=== OTIMIZANDO THRESHOLD ===")
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        best_f1 = 0
        for thresh in self.threshold_range:
            y_pred = (y_val_proba >= thresh).astype(int)
            f1 = f1_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred)
            logger.info(f"Threshold {thresh:.1f}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                self.best_threshold = thresh
        
        logger.info(f"Melhor threshold: {self.best_threshold} (F1={best_f1:.4f})")
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Avalia modelo em um dataset"""
        if threshold is None:
            threshold = self.best_threshold
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            "auc": roc_auc_score(y, y_pred_proba),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }
        
        logger.info(f"\n{dataset_name.upper()} (threshold={threshold}):")
        logger.info(f"  AUC-ROC:   {metrics['auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def check_overfitting(self, metrics_train: Dict, metrics_val: Dict):
        """Template step: Verifica overfitting"""
        logger.info("=== OVERFITTING CHECK ===")
        delta_auc = metrics_train["auc"] - metrics_val["auc"]
        delta_f1 = metrics_train["f1"] - metrics_val["f1"]
        
        logger.info(f"Delta AUC (train-val): {delta_auc:+.4f}")
        logger.info(f"Delta F1 (train-val):  {delta_f1:+.4f}")
        
        if delta_auc > 0.05 or delta_f1 > 0.05:
            logger.warning("⚠️  OVERFITTING DETECTADO")
        else:
            logger.info("✓ Generalização OK")
    
    def show_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Template step: Exibe matriz de confusão"""
        logger.info("=== CONFUSION MATRIX (TEST) ===")
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        logger.info(f"FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        logger.info(f"\nTotal cancelamentos: {y_test.sum()}")
        logger.info(f"Detectados (TP): {cm[1,1]} ({cm[1,1]/y_test.sum()*100:.1f}%)")
        logger.info(f"Perdidos (FN):   {cm[1,0]} ({cm[1,0]/y_test.sum()*100:.1f}%)")
    
    def show_feature_importance(self, top_k: int = 15):
        """Template step: Exibe feature importance"""
        logger.info(f"=== FEATURE IMPORTANCE (Top {top_k}) ===")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info(f"\n{'Rank':<6} {'Feature':<40} {'Importance':>12}")
        logger.info("-" * 60)
        for i, idx in enumerate(indices[:top_k], 1):
            logger.info(f"{i:<6} {self.feature_names[idx]:<40} {importances[idx]:>12.6f}")
    
    def save_model(self, path: Optional[Path] = None):
        """Template step: Salva modelo"""
        if path is None:
            path = config.models.production / "gradient_boosting_churn.joblib"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")
    
    def save_results(self, metrics_train: Dict, metrics_val: Dict, metrics_test: Dict):
        """Template step: Salva resultados em CSV"""
        results_df = pd.DataFrame({
            'Dataset': ['Train', 'Validation', 'Test'],
            'AUC_ROC': [metrics_train["auc"], metrics_val["auc"], metrics_test["auc"]],
            'Precision': [metrics_train["precision"], metrics_val["precision"], metrics_test["precision"]],
            'Recall': [metrics_train["recall"], metrics_val["recall"], metrics_test["recall"]],
            'F1_Score': [metrics_train["f1"], metrics_val["f1"], metrics_test["f1"]],
        })
        
        results_path = config.outputs / "metrics" / "gb_results.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        logger.info(f"Resultados salvos: {results_path}")
    
    def save_feature_importance(self):
        """Template step: Salva feature importance"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'Feature': [self.feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        importance_path = config.outputs / "metrics" / "gb_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance salvo: {importance_path}")
    
    def train(self):
        """
        Template Method: Executa pipeline completo de treinamento.
        Define ordem fixa dos steps, cada step pode ser customizado.
        """
        logger.info("=== TREINAMENTO GRADIENT BOOSTING ===")
        
        # 1. Carregar dados
        df_train, df_val, df_test = self.load_data()
        
        # 2. Preparar features
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_features(
            df_train, df_val, df_test
        )
        
        # 3. Treinar modelo
        logger.info("─" * 70)
        logger.info("[3/10] 🤖 Treinando modelo...")
        logger.info("─" * 70)
        self.fit_model(X_train, y_train)
        
        # 4. Otimizar threshold
        logger.info("─" * 70)
        logger.info("[4/10] 🎚️  Otimizando threshold...")
        logger.info("─" * 70)
        self.optimize_threshold_on_validation(X_val, y_val)
        
        # 5. Avaliar em todos os datasets
        logger.info("─" * 70)
        logger.info("[5/10] 📊 Avaliando performance...")
        logger.info("─" * 70)
        metrics_train = self.evaluate(X_train, y_train, "train")
        metrics_val = self.evaluate(X_val, y_val, "validation")
        metrics_test = self.evaluate(X_test, y_test, "test")
        
        # 6. Verificar overfitting
        logger.info("─" * 70)
        logger.info("[6/10] 🔍 Verificando overfitting...")
        logger.info("─" * 70)
        self.check_overfitting(metrics_train, metrics_val)
        
        # 7. Confusion matrix
        logger.info("─" * 70)
        logger.info("[7/10] 📈 Matriz de confusão...")
        logger.info("─" * 70)
        self.show_confusion_matrix(X_test, y_test)
        
        # 8. Feature importance
        logger.info("─" * 70)
        logger.info("[8/10] 🔬 Feature importance...")
        logger.info("─" * 70)
        self.show_feature_importance(top_k=15)
        
        # 9. Salvar modelo
        logger.info("─" * 70)
        logger.info("[9/10] 💾 Salvando modelo...")
        logger.info("─" * 70)
        self.save_model()
        
        # 10. Salvar resultados
        logger.info("─" * 70)
        logger.info("[10/10] 📁 Salvando métricas...")
        logger.info("─" * 70)
        self.save_results(metrics_train, metrics_val, metrics_test)
        self.save_feature_importance()
        
        logger.info("=" * 70)
        logger.info("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 70)
        
        return {
            "train": metrics_train,
            "validation": metrics_val,
            "test": metrics_test,
            "threshold": self.best_threshold,
        }


def train_gradient_boosting() -> Dict:
    """
    Função utilitária: Treina modelo Gradient Boosting com configurações padrão.
    """
    trainer = GradientBoostingTrainer()
    results = trainer.train()
    return results
