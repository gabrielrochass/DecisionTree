# Classificação da Qualidade do Vinho Branco usando Árvore de Decisão

Este projeto visa utilizar técnicas de aprendizado de máquina para classificar a qualidade do vinho branco com base em suas características. Uma Árvore de Decisão é empregada como modelo de classificação.

## Descrição

Este repositório contém o código-fonte em Python para treinar e avaliar um modelo de Árvore de Decisão na tarefa de classificação da qualidade do vinho branco. O código realiza as seguintes etapas:

1. **Pré-processamento dos Dados:** Carrega o conjunto de dados `winequality-white.csv`, prepara os dados e os divide em conjuntos de treinamento e teste.

2. **Treinamento do Modelo:** Utiliza o conjunto de treinamento para treinar o modelo de Árvore de Decisão, com ajuste de parâmetros para evitar overfitting e underfitting.

3. **Avaliação do Modelo:** Avalia o desempenho do modelo treinado utilizando o conjunto de teste e imprime a acurácia.

4. **Visualização da Árvore de Decisão:** Plota a estrutura da árvore de decisão resultante para interpretabilidade.

5. **Análise de Overfitting e Underfitting:** Explora os efeitos de ajustar os parâmetros do modelo na capacidade de generalização.

## Requisitos

- Python 3.11.1
- Pandas
- NumPy
- scikit-learn
- Matplotlib

## Instalação

1. Clone este repositório em seu ambiente local:

`git clone https://github.com/seu-usuario/DecisionTree.git`

2. Navegue até o diretório do projeto:

`cd classificacao-vinho-branco`

3. Instale as dependências utilizando o pip:

`pip install -r requirements.txt`

## Como Usar

Execute o script `wine_quality_classification.py` para treinar o modelo e visualizar os resultados:

`python wine_quality_classification.py`
