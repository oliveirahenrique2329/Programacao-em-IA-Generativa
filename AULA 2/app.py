import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# 1. Preparação dos Dados
# Criando o DataFrame fornecido
estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# Separando as variáveis preditoras (features) e o alvo (target)
# No TensorFlow, é uma boa prática usar arrays numpy do tipo float32
X = np.array(estudos['horas'], dtype=float)
y = np.array(estudos['notas'], dtype=float)

print("Dados de entrada (Horas):", X)
print("Dados de saída (Notas):", y)
print("-" * 40)

# 2. Definição do Modelo (Arquitetura)
# O Sequential agrupa uma pilha de camadas lineares em sequência.
# Como é uma regressão linear simples, usamos uma única camada densa (Dense) 
# com 1 neurônio e 1 dado de entrada (input_shape=[1]).
modelo = Sequential([
    Dense(units=1, input_shape=[1])
])

# 3. Compilação do Modelo
# Precisamos definir o otimizador (como o modelo ajusta os pesos) e a função de perda (erro).
# 'sgd' = Stochastic Gradient Descent (Gradiente Descendente Estocástico)
# 'mean_squared_error' = Erro Quadrático Médio, ideal para problemas de regressão.
modelo.compile(
    optimizer='sgd', 
    loss='mean_squared_error'
)

# Exibindo o resumo da arquitetura do modelo
modelo.summary()
print("-" * 40)

# 4. Treinamento (Fit)
# Treinamos o modelo por 500 épocas (voltas completas nos dados) 
# verbose=0 oculta a barra de progresso para deixar o output mais limpo.
print("Treinando o modelo...")
historico = modelo.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído!")
print("-" * 40)

# 5. Realizando Predições
# Vamos testar o modelo prevendo a nota para um aluno que estudou 6 horas
horas_teste = np.array([6.0], dtype=float)
previsao = modelo.predict(horas_teste)

print(f"Previsão para {horas_teste[0]} horas de estudo:")
print(f"Nota prevista: {previsao[0][0]:.2f}")