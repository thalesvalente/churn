# Gerenciamento de Modelos - Experiments vs Production

## 📁 Estrutura de Diretórios

```
models/
├─ experiments/          # Modelos treinados automaticamente pelo pipeline
│  ├─ gradient_boosting_churn.joblib
│  ├─ tab_n1_semantics.joblib
│  ├─ tab_n2_semantics.joblib
│  └─ tab_n3_semantics.joblib
│
└─ production/          # Modelos validados e prontos para uso em produção
   └─ (vazio inicialmente - mover manualmente)
```

## 🎯 Filosofia

- **Experiments:** Modelos são salvos automaticamente aqui durante o treinamento
- **Production:** Apenas modelos validados e aprovados devem estar aqui
- **Controle Manual:** Mover para produção é uma decisão consciente e documentada

## ✅ Processo de Validação e Promoção

### 1. Treinar Modelo

```bash
python scripts/run_complete_pipeline.py
```

Isso gera:
- ✅ Modelos em `models/experiments/`
- ✅ Métricas em `outputs/metrics/gb_results.csv`
- ✅ Relatório técnico em `outputs/reports/`

### 2. Validar Resultados

Revisar o relatório técnico gerado:

```markdown
## Critérios de Validação

- [ ] F1-Score ≥ 95% no conjunto de teste
- [ ] AUC-ROC ≥ 99% no conjunto de teste
- [ ] Sem sinais de overfitting (diferença Train-Val < 3%)
- [ ] Feature importance coerente (TICKET_MEDIO como top feature)
- [ ] Reprodutibilidade confirmada (random_state=42)
- [ ] Logs completos sem erros
- [ ] Coluna SITUACAO removida (prevenção de data leakage)
```

### 3. Mover para Produção (Manualmente)

**⚠️ ATENÇÃO:** Este passo deve ser feito manualmente após validação

#### Windows (PowerShell):

```powershell
# Criar backup se já existir modelo em produção
if (Test-Path models/production/gradient_boosting_churn.joblib) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item models/production/*.joblib models/production/backup_$timestamp/
}

# Mover modelos validados
Copy-Item models/experiments/gradient_boosting_churn.joblib models/production/
Copy-Item models/experiments/tab_n*_semantics.joblib models/production/

Write-Host "✅ Modelos promovidos para produção!" -ForegroundColor Green
```

#### Linux/Mac:

```bash
# Criar backup se já existir modelo em produção
if [ -f models/production/gradient_boosting_churn.joblib ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p models/production/backup_$timestamp
    cp models/production/*.joblib models/production/backup_$timestamp/
fi

# Mover modelos validados
cp models/experiments/gradient_boosting_churn.joblib models/production/
cp models/experiments/tab_n*_semantics.joblib models/production/

echo "✅ Modelos promovidos para produção!"
```

### 4. Documentar Mudança

Criar registro em `models/production/CHANGELOG.md`:

```markdown
## [2025-11-19] - Modelo v1.0.0

### Métricas
- F1-Score (Test): 95.46%
- AUC-ROC (Test): 99.41%
- Precision: 95.99%
- Recall: 94.95%

### Mudanças
- Primeira versão de produção
- Remoção de coluna SITUACAO (data leakage)
- Pipeline completo implementado
- 37 features engenheiradas

### Validação
- ✅ Métricas acima do threshold
- ✅ Sem overfitting
- ✅ Reprodutível (random_state=42)

### Treinado por
- Script: run_complete_pipeline.py
- Data: 2025-11-19
- Tempo: 14.3 minutos
- Relatório: outputs/reports/RELATORIO_TECNICO_20251119_094254.md
```

## 🚫 O Que NÃO Fazer

❌ **Não commitar modelos diretamente em production**
- Use experiments primeiro
- Valide antes de promover

❌ **Não sobrescrever modelos em produção sem backup**
- Sempre criar backup com timestamp
- Manter histórico de versões

❌ **Não pular validação**
- Sempre revisar métricas
- Sempre gerar relatório técnico
- Sempre documentar mudanças

## 📊 Versionamento de Modelos

Sugestão de nomenclatura para backups:

```
models/production/
├─ gradient_boosting_churn.joblib         # Versão atual
├─ backup_20251119_094254/                 # Backup com timestamp
│  ├─ gradient_boosting_churn.joblib
│  └─ tab_n*_semantics.joblib
└─ CHANGELOG.md                            # Histórico de mudanças
```

## 🔄 Rollback

Se precisar reverter para versão anterior:

```powershell
# Windows
$backup = "backup_20251119_094254"
Copy-Item models/production/$backup/*.joblib models/production/
Write-Host "✅ Rollback completo para $backup"
```

```bash
# Linux/Mac
backup="backup_20251119_094254"
cp models/production/$backup/*.joblib models/production/
echo "✅ Rollback completo para $backup"
```

## 📝 Checklist de Promoção

Antes de mover para produção, confirme:

- [ ] Pipeline executado sem erros
- [ ] Relatório técnico revisado
- [ ] Métricas validadas (F1 ≥ 95%)
- [ ] Backup criado (se modelo anterior existe)
- [ ] CHANGELOG.md atualizado
- [ ] Commit no git com tag de versão
- [ ] Equipe notificada da mudança
- [ ] Documentação atualizada

---

**Última atualização:** 19/11/2025
