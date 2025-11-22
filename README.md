# Churn Prediction - Telecomunicações

**POC de Machine Learning para predição de cancelamento de clientes**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.0+-yellow.svg)](https://huggingface.co/transformers/)

## 📊 Resultados do Modelo

**Gradient Boosting Classifier** - Métricas no dataset de teste:

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **95.5%** |
| AUC-ROC | 99.4% |
| Precision | 96.0% |
| Recall | 94.9% |

- **263 de 277 cancelamentos detectados** (94.9% recall)
- **11 falsos positivos** (apenas 4.0% falso alarme)
- **Threshold otimizado: 0.2** (calibrado em validation)

---

## 🚀 Quick Start

### 1. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 2. Preparar dados

Coloque o dataset original em `data/raw/`:
- `dataset.xlsx` (433.588 registros)

O pipeline criará automaticamente os splits train/validation/test (80/10/10)

### 3. Executar pipeline completo

```powershell
python scripts/run_complete_pipeline.py
```

Ou executar steps individualmente:

```powershell
python scripts/00_prepare_data.py        # Preparação dos dados crus
python scripts/01_process_semantics.py   # Clustering semântico
python scripts/02_engineer_features.py   # Feature engineering  
python scripts/03_train_model.py         # Treinamento
```

### 4. Resultados Gerados

Após execução:
- ✅ **Modelos**: `models/experiments/` (validar antes de mover para production)
- ✅ **Métricas**: `outputs/metrics/gb_results.csv`
- ✅ **Relatório Técnico**: `outputs/reports/RELATORIO_TECNICO_*.md` (gerado automaticamente)

---

## 🧠 Metodologia

1. **Preparação de Dados** (Split estratificado 80/10/10, criação de TARGET)
2. **Clustering Semântico** (Sentence Transformers + K-Means)
   - TAB_N1: 6 clusters
   - TAB_N2: 12 clusters  
   - TAB_N3: 18 clusters
3. **Feature Engineering** (37 features: contadores, sentiment, escalation, flags)
4. **Gradient Boosting** (threshold otimizado = 0.2)
5. **Relatório Técnico Automático** (gerado ao final)

**Top 5 Features**:
- `TICKET_MEDIO` (86.4%) - Valor médio de ticket do cliente
- `IDADE_APROX` (5.9%) - Idade aproximada do cliente
- `MESES` (4.4%) - Tempo como cliente (meses)
- `sent_score_std` (1.1%) - Desvio padrão do sentimento
- `count_tab_n3_info_cancelamento` (0.4%) - Menções de cancelamento

---

## 🏗️ Estrutura

```
em/
├── data/              # Dados (raw → processed → features)
├── src/               # Código de produção
│   ├── preprocessing/ # Semantic + Feature Engineering
│   ├── training/      # Gradient Boosting Trainer
│   └── utils/         # Logger, config, report_generator
├── scripts/           # Entry points (00, 01, 02, 03, run_complete_pipeline)
├── models/
│   ├── experiments/   # Modelos gerados pelo pipeline (validar primeiro)
│   └── production/    # Modelos aprovados (mover manualmente)
├── outputs/           # Métricas, logs, reports (relatório técnico automático)
└── docs/              # Documentação completa
```

---

## 🛠️ Design Patterns

- **Singleton**: Config, Logger
- **Factory**: Criar estratégias de clustering
- **Strategy**: TAB_N1/N2/N3 com configs distintas
- **Builder**: Feature engineering incremental
- **Template Method**: Pipeline de treinamento

---

## ⚠️ Informações Importantes

### Natureza do Experimento

Este experimento foi realizado como **POC inicial** para conhecer os dados e validar a viabilidade técnica de predição de churn. Os resultados apresentados (F1-Score de 95.5%) devem ser interpretados com cautela devido a **problemas metodológicos críticos** que precisam ser resolvidos antes de qualquer aplicação em produção.

### Vazamento de Dados (Data Leakage)

**Problema Identificado**: Os dados **não foram divididos por cliente** durante o split train/validation/test. Como a maior parte dos clientes possui **ticket médio único** (característica identificadora), o modelo provavelmente aprendeu padrões específicos de clientes que aparecem em múltiplos registros distribuídos entre treino e teste.

**O que é Data Leakage?**

- Ocorre quando informações do conjunto de teste "vazam" para o treinamento
- No nosso caso: mesmo cliente aparece em train e test com features similares
- O modelo memoriza padrões de clientes específicos ao invés de generalizar
- Métricas ficam artificialmente infladas e não representam desempenho real

**Impacto**:

- ✅ Métricas atuais são **otimistas demais**
- ❌ Desempenho real em clientes novos será **significativamente inferior**
- ❌ Modelo atual **não é confiável** para produção

### Próximos Passos para Mitigação

Para resolver o problema de vazamento e construir um modelo confiável:

1. **Split por Cliente** (Crítico)
   - Garantir que cada cliente apareça em apenas um conjunto (train/val/test)
   - Implementar split estratificado mantendo distribuição de churn
   - Validar que não há sobreposição de clientes entre conjuntos

2. **Reavaliar Feature Engineering**
   - Remover ou transformar features que identifiquem clientes únicos
   - Revisar `TICKET_MEDIO` (feature mais importante - pode ser identificador)
   - Focar em features agregadas temporais e comportamentais

3. **Validação Temporal**
   - Considerar split temporal (ex: treinar em 2023, testar em 2024)
   - Simular cenário real: prever churn de clientes futuros
   - Avaliar degradação do modelo ao longo do tempo

4. **Cross-Validation Estratificada por Cliente**
   - Usar K-Fold com agrupamento por cliente
   - Obter estimativa mais realista de desempenho
   - Identificar overfitting e instabilidade do modelo

5. **Benchmark Conservador**
   - Comparar com baseline simples (regressão logística)
   - Documentar queda esperada nas métricas
   - Estabelecer threshold de aceitação realista

⚠️ **Recomendação**: Não utilizar este modelo em produção antes de refazer o experimento com split correto por cliente.

---

## 🔄 Promoção para Produção

Modelos são salvos em `models/experiments/` por padrão. Para produção:

1. **Validar métricas** no relatório técnico (`outputs/reports/`)
2. **Mover manualmente** para `models/production/` (ver `models/README.md`)
3. **Documentar mudança** em `models/production/CHANGELOG.md`

⚠️ **Nunca mova modelos para production sem validação manual!**

---

## 📚 Documentação

- `README.md` (este arquivo) - Quick start e visão geral
- `outputs/reports/RELATORIO_TECNICO_*.md` - Relatórios técnicos automáticos (gerados a cada execução)
- `docs/PIPELINE_COMPLETO.md` - Documentação detalhada do pipeline
- `docs/GUIA_EXECUCAO.md` - Guia de execução e troubleshooting
- `models/README.md` - Processo de promoção para produção
- `src/config.py` - Configurações customizáveis
