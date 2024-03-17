import pandas as pd
import numpy as np
# Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from time import time
# gráfico
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# carrega o dataset
whitewine = pd.read_csv(r"/work/winequality-white.csv", delimiter=";")

X = whitewine.drop(['quality'], axis = 1) 
y = whitewine['quality']

# divide o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,test_size = 0.3,random_state = 1111)

# inicia o tempo
t0 = time()

# cria o classificador de DT
classifier = DecisionTreeClassifier(max_depth=10)

# treina o classificador
classifier = classifier.fit(X_train,y_train)

# prevê a resposta
y_pred = classifier.predict(X_test)
acc_dt = round(classifier.score(X_test,y_test) * 100, 2)


# printa acurácia
print ("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred)*100,"%")
print('Training time', round(time() - t0, 3), 's')

# plota a árvore
# normaliza os dados
scaler = StandardScaler() # cria o objeto de normalização
X_scaled = scaler.fit_transform(X) # normaliza de fato

# plota as fronteiras de decisão
plt.figure(figsize=(15, 10)) # tamanho do gráfico
plot_colors = "ryb" # cores do gráfico
plot_step = 0.02 # passo do gráfico -> quanto menor, mais detalhado

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]): # itera sobre as combinações de pares de características
    # selecionar apenas as duas características correspondentes
    X_train_pair = X_train[:, pair] 
    X_test_pair = X_test[:, pair]

    # treinar o classificador para aquelas duas características
    clf.fit(X_train_pair, y_train)

    # calcula limites do gráfico -> mínimo e máximo definem os limites
    x_min, x_max = X_train_pair[:, 0].min() - 1, X_train_pair[:, 0].max() + 1
    y_min, y_max = X_train_pair[:, 1].min() - 1, X_train_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # prevê para cada ponto na malha
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # ravel: transforma a matriz em um vetor
    Z = Z.reshape(xx.shape) # reshape: remodela o vetor em uma matriz

    # Plotar o contorno e os pontos de treinamento
    plt.subplot(2, 3, pairidx + 1) # cria um subplot
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8) # cria o contorno
    plt.scatter(X_train_pair[:, 0], X_train_pair[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolor='k') # plota os pontos de treinamento
    plt.xlabel(whitewine.columns[pair[0]]) # nomeia o eixo x
    plt.ylabel(whitewine.columns[pair[1]]) # nomeia o eixo y
    plt.title(f"Decision boundary for feature pair {pair}") # título do gráfico

plt.tight_layout() # ajusta o layout
plt.show() # mostra o gráfico

# underfitting: desempenho do modelo já é ruim no treinamento e permanece assim no conjunto de teste
# modelo não consegue encontrar relações entre as variáveis
# modelo descartável

# novo classificador com max_depth menor para induzir underfitting
classifier_underfitting = DecisionTreeClassifier(max_depth=2)

# treina o modelo
classifier_underfitting.fit(X_train, y_train)

# prevê os resultados
y_pred_underfitting = classifier_underfitting.predict(X_test)

# calcula e printa o desempenho do modelo
accuracy_underfitting = metrics.accuracy_score(y_test, y_pred_underfitting) # quanto menor, maior indicativo de underfitting, já foi ruim desde sempre
print("Accuracy (Underfitting):", accuracy_underfitting, "%")

# overfitting: desempenho bom no conjunto de treinamento e ruim no conjunto real
# o modelo apenas decorou o que precisava fazer
# com regras decoradas, o modelo tenta replicar o que fez no conjunto de treinamento, afetando o desempenho e a validade da informação
# modelo sem capacidade de generalização
 
# aumentando profundidade máxima
classifier_overfit = DecisionTreeClassifier(max_depth=25)

# reduzindo o tamanho do conjunto de treinamento -> 5% de treinamento e 95% de teste
X_train_overfit, X_test_overfit, y_train_overfit, y_test_overfit = train_test_split(X, y, test_size=0.95, random_state=42)

# treinando o novo classificador com o conjunto de treinamento reduzido
classifier_overfit.fit(X_train_overfit, y_train_overfit)

# realizando a previsão nos dados de teste
y_pred_overfit = classifier_overfit.predict(X_test_overfit)

# calculando a nova acurácia
acc_overfit = round(classifier_overfit.score(X_test_overfit, y_test_overfit) * 100, 2)

# printa acurácia -> quanto menor, maior indicativo de overfitting, pois lidou pior com os dados de teste
print("Decision Tree Accuracy (Overfitting):", acc_overfit, "%")
