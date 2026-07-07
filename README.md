# Classificação de Qualidade de Vinhos — Tech Challenge Fase 2

Modelo de classificação binária para prever se um vinho é de **alta qualidade** (nota ≥ 7) ou
**baixa/média qualidade** (nota < 7) a partir de variáveis físico-químicas, usando o dataset
[Wine Quality Dataset](https://www.kaggle.com/datasets) (Kaggle).

## Contexto

A avaliação tradicional de qualidade de vinho é feita por especialistas via análise sensorial —
processo subjetivo e dependente da experiência do avaliador. Este projeto avalia se variáveis
físico-químicas mensuráveis durante a produção (acidez, teor alcoólico, densidade, dióxido de
enxofre etc.) conseguem prever essa classificação, como ferramenta de apoio à decisão para
enólogos e produtores.

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/                # Base de dados utilizada (WineQT.csv)
├── notebooks/           # Notebook com a análise exploratória e modelagem
├── src/                 # Scripts auxiliares (se aplicável)
├── results/             # Gráficos, matrizes de confusão e planilha de classificação exportada
├── requirements.txt     # Bibliotecas utilizadas
└── README.md            # Este arquivo
```

## Metodologia

O pipeline segue as etapas abaixo, todas documentadas no notebook em `notebooks/`:

### 1. Compreensão do problema e EDA
- Transformação da variável `quality` em alvo binário (`target`: 1 se ≥ 7, senão 0).
- Análise de distribuição, correlação com a variável alvo (com justificativa físico-química de
  cada correlação relevante) e verificação de dados faltantes.
- Detecção de outliers via regra do IQR, com decisão registrada de mantê-los (não há evidência de
  erro de medição, e os modelos baseados em árvore são robustos a esses casos).

### 2. Pré-processamento e feature engineering
- Padronização (`StandardScaler`) para o modelo linear (Regressão Logística).
- Três variáveis derivadas: `acidez_total`, `razao_so2_livre` e `alcool_por_densidade` — esta
  última se revelou a variável de maior importância no modelo final.

### 3. Divisão dos dados
- Split estratificado em **treino (70%) / validação (15%) / teste (15%)**, com o conjunto de
  teste reservado exclusivamente para a avaliação final.

### 4. Tratamento do desbalanceamento
- Como a classe "alta qualidade" é minoritária (~14% da base), três estratégias foram comparadas
  via validação cruzada estratificada (5 folds), usando `average_precision` (AUC-PR) como
  métrica: `RandomOverSampler`, `SMOTE` e `class_weight='balanced'`.

### 5. Comparação de modelos
- Quatro algoritmos comparados via validação cruzada: **Random Forest, Regressão Logística,
  Gradient Boosting e XGBoost**, além de um baseline (`DummyClassifier`) para contextualizar as
  métricas.
- Ajuste de hiperparâmetros do modelo vencedor via `GridSearchCV`.

### 6. Calibração de limiar e avaliação
- Os limiares de decisão (para classificar como "alta qualidade" vs. "potencialmente alta
  qualidade") foram calibrados observando a curva Precision-Recall **no conjunto de validação**.
- O conjunto de teste foi usado uma única vez, no final, para reportar as métricas oficiais —
  evitando o vazamento de informação que ocorreria se o limiar fosse ajustado olhando o teste.

### 7. Interpretação dos resultados
- Extração da importância das variáveis do modelo final, com discussão das implicações práticas
  para o processo produtivo (ex.: controle de acidez volátil como prevenção de defeito, não como
  meta de otimização).

## Resultados principais

| Item | Resultado |
|---|---|
| Melhor estratégia de balanceamento | RandomOverSampler |
| Melhor modelo | Random Forest |
| AUC-PR (validação) | ~0.77 |
| AUC-PR (teste, avaliação final) | ~0.77 |
| Variável mais importante | `alcool_por_densidade` (feature derivada) |

Métricas completas (precisão, recall, f1-score e matrizes de confusão) estão no notebook e na
pasta `results/`.

## Limitações

- O modelo captura **associação**, não causalidade: alterar uma variável isoladamente no processo
  produtivo não garante a mudança correspondente na nota prevista.
- O dataset não inclui fatores como uva, terroir e clima da safra, que também influenciam a
  avaliação sensorial.
- A classe "alta qualidade" é pequena (~160 amostras), o que limita a robustez estatística das
  métricas mesmo com validação cruzada.

## Como reproduzir

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd wine-quality-classification

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Abrir o notebook
jupyter notebook notebooks/Tech_challenge_2_vinhos_corrigido.ipynb
```

## Bibliotecas utilizadas

Ver `requirements.txt` para as versões exatas. Principais: `pandas`, `numpy`, `scikit-learn`,
`imbalanced-learn`, `xgboost`, `seaborn`, `matplotlib`, `openpyxl`.
