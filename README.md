# Projeto de Previsão de Rotatividade de Funcionários (Churn Prediction)

Este projeto tem como objetivo prever a saída de funcionários de uma empresa com base em dados históricos. Além disso, buscamos identificar os principais fatores que levam ao desligamento, permitindo a adoção de estratégias de retenção mais eficazes.

---

## Tema do Projeto

**Análise e Previsão de Rotatividade de Funcionários (Employee Attrition)**
Classificação supervisionada utilizando algoritmos de Machine Learning para prever se um funcionário irá sair da empresa.

---

## Ferramentas Utilizadas

**Linguagens:**

* Python

**Bibliotecas Python:**

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* xgboost

**Ferramentas e Ambientes:**

* Jupyter Notebook
* Excel
* Power BI
* Git & GitHub

**Conceitos aplicados:**

* Machine Learning (Classificação)
* Engenharia de Atributos
* Análise Exploratória de Dados (EDA)
* Data Cleaning e Data Preparation
* Balanceamento de Dados (SMOTE)
* Normalização e Codificação
* Avaliação de Modelos (Accuracy, Recall, Precision, F1-score)
* Interpretação de Importância de Variáveis
* Visualização de Dados

---

## O que foi feito (e o que aprendi)

* Realizamos uma análise completa dos dados de Recursos Humanos da empresa.
* Aplicamos técnicas de pré-processamento, como codificação de variáveis categóricas e normalização de dados.
* Utilizamos o SMOTE para balancear a base de dados.
* Treinamos e avaliamos diversos modelos, incluindo:

  * Regressão Logística
  * Árvore de Decisão
  * Random Forest
  * XGBoost
* Identificamos os principais fatores que levam ao desligamento dos funcionários.
* Comparamos o desempenho dos modelos e construímos gráficos de importância das variáveis.
* Aprendi a interpretar modelos além da acurácia, analisando métricas como F1-score e Recall, especialmente nos casos de classe minoritária.
* Ao final, montei um relatório conclusivo e preparei o projeto para publicação no GitHub.

---

## Conclusões: Por que os funcionários estão saindo da empresa?

Com base na análise exploratória e nos modelos de machine learning, especialmente o XGBoost, identificamos os principais fatores que influenciam a rotatividade dos colaboradores. As variáveis com maior ganho (gain) no modelo revelam padrões importantes que ajudam a entender o comportamento dos desligamentos.

### Principais fatores de desligamento

* **StockOptionLevel (Nível de opção de ações)** – **8.37%**
  Funcionários sem opção de ações ou com níveis mais baixos têm maior tendência a sair.

* **Departamento:**

  * **Sales (Vendas)** – **8.11%**
  * **Research & Development (P\&D)** – **7.15%**
    Departamentos estratégicos com maior rotatividade.

* **Cargo (JobRole):**

  * **Sales Representative** – **6.37%**
  * **Human Resources** – **6.26%**
    Cargos que enfrentam maior insatisfação ou pressão.

* **JobLevel e JobInvolvement**
  Colaboradores em níveis hierárquicos mais baixos e com pouco envolvimento tendem a sair mais.

* **Satisfação no trabalho**
  Variáveis como **JobSatisfaction** e **RelationshipSatisfaction** também foram influentes.

### Conclusão Final

Os principais motivos para desligamento estão ligados a:

* Baixo incentivo financeiro ou de longo prazo
* Falta de engajamento e satisfação no trabalho
* Pressões em áreas estratégicas e cargos específicos
* Ausência de plano de carreira claro

> **Nosso modelo não apenas previu com boa precisão quem pode sair, como também forneceu insights valiosos para que a empresa atue preventivamente com base em dados.**
