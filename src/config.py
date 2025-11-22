"""
Configuração centralizada do projeto.
Pattern: Singleton para garantir configuração única.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class DataPaths:
    """Paths para dados (Pattern: Value Object)"""
    raw: Path
    processed: Path
    features: Path
    
    @classmethod
    def from_base(cls, base_path: Path) -> "DataPaths":
        return cls(
            raw=base_path / "raw",
            processed=base_path / "processed",
            features=base_path / "features"
        )


@dataclass
class ModelPaths:
    """Paths para modelos"""
    production: Path
    experiments: Path
    
    @classmethod
    def from_base(cls, base_path: Path) -> "ModelPaths":
        return cls(
            production=base_path / "production",
            experiments=base_path / "experiments"
        )


@dataclass
class SemanticConfig:
    """Configuração de clustering semântico"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = None
    batch_size: int = None
    tab_configs: Dict[str, Dict] = None
    
    def __post_init__(self):
        # Auto-detect GPU
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"🖥️  GPU detectada: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU only'}")
            except:
                self.device = "cpu"
        
        # Ajustar batch_size baseado no device (igual ao experimento anterior)
        if self.batch_size is None:
            self.batch_size = 64 if self.device == "cuda" else 32
        
        # Configurar tabs
        if self.tab_configs is None:
            self.tab_configs = {
                "TAB_N1": {
                    "n_clusters": 6,  # Idêntico ao experimento original
                    "min_cluster_size": 50,
                },
                "TAB_N2": {
                    "n_clusters": 12,
                    "min_cluster_size": 30,
                },
                "TAB_N3": {
                    "n_clusters": 18,
                    "min_cluster_size": 20,
                }
            }


@dataclass
class FeatureEngineeringConfig:
    """Configuração de feature engineering"""
    clusters_criticos: Dict[str, Dict[str, List[str]]] = None
    
    def __post_init__(self):
        if self.clusters_criticos is None:
            self.clusters_criticos = {
                "TAB_N1": {
                    "reclamacao": ["reclamação, sugestão, elogios, denuncia"],
                    "cobranca": ["ativo cobrança, cobrança, ativo, gestão base"],
                },
                "TAB_N2": {
                    "sem_conexao": ["sem conexão, conexão, sem, debitos aberto"],
                    "cobranca": ["cobrança, celula cobrança, celula"],
                },
                "TAB_N3": {
                    "cobranca_inadimplencia": ["cobrança inadimplencia, inadimplencia, cobrança, ligação cobrança"],
                    "sem_sinal": ["sem sinal, sem, sinal, lentidão"],
                    "bloqueado": ["bloqueado, bloqueado debitos, debitos, debitos aberto"],
                    "cancelamento_ura": ["ura, cancelamento, cancelamento ura, tds ptos"],
                    "nao_navega": ["não navega, navega, não, adesão"],
                    "oscilando": ["oscilando"],
                    "info_cancelamento": ["informações, sobre, cancelamento, sobre cancelamento"],
                }
            }


@dataclass
class TrainingConfig:
    """Configuração de treinamento"""
    model_type: str = "gradient_boosting"
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    random_state: int = 42
    
    threshold_optimization: bool = True
    threshold_range: List[float] = None
    
    def __post_init__(self):
        if self.threshold_range is None:
            self.threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


class Config:
    """
    Configuração global do projeto (Pattern: Singleton).
    Garante uma única instância de configuração em toda aplicação.
    """
    _instance: Optional["Config"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Base paths
        self.project_root = Path(__file__).parent.parent
        self.data = DataPaths.from_base(self.project_root / "data")
        self.models = ModelPaths.from_base(self.project_root / "models")
        self.outputs = self.project_root / "outputs"
        
        # Configs de módulos
        self.semantic = SemanticConfig()
        self.feature_engineering = FeatureEngineeringConfig()
        self.training = TrainingConfig()
        
        # Colunas a remover (data leakage / não informativas)
        self.columns_to_remove = [
            "TARGET", "TAB_N1", "TAB_N2", "TAB_N3",
            "TAB_N1_GROUP", "TAB_N2_GROUP", "TAB_N3_GROUP",
            "TAB_N1_SENT_LABEL", "TAB_N2_SENT_LABEL", "TAB_N3_SENT_LABEL",
            "DESCRICAO_OS", "PROBLEMA_ENCONTRADO", "STATUS_OS", "OBS_ABERTURA",
            "CANAL", "GENERO", "SERVICO", "PROFISSAO",
            "CIDADE_x", "BAIRRO_x", "CIDADE_y", "BAIRRO_y",
            "DATA_REGISTRO", "HORA_REGISTRO", "ULTIMO_CORTE_INAD",
            "SITUACAO", "LIGACAO_RECORRENTE", "GEROU_OS"
        ]
        
        self._initialized = True
    
    def ensure_paths_exist(self):
        """Cria diretórios necessários"""
        for path in [
            self.data.raw, self.data.processed, self.data.features,
            self.models.production, self.models.experiments,
            self.outputs / "metrics", self.outputs / "reports"
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Instância global
config = Config()
