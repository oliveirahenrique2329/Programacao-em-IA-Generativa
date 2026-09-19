import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1. Preparação dos Dados
# Dados informados: Horas de estudo (entrada) e Desempenho/Nota (saída)
df = pd.DataFrame(
    {"horas_estudo": [1, 2, 4, 6, 8, 10], "nota": [1, 2, 3, 5, 8, 10]}
)

# O TensorFlow espera arrays do NumPy com tipos numéricos definidos (ex: float32)
X = np.array(df["horas_estudo"], dtype=float)
y = np.array(df["nota"], dtype=float)

# 2. Construção da Rede Neural (Modelo)
# Usamos o Sequential para empilhar camadas em sequência.
# A camada Dense(units=1) representa o nosso neurônio da regressão linear: y = w*x + b
model = Sequential(
    [
        layers.Dense(
            units=1, input_shape=[1]
        )  # units=1 (1 neurônio), input_shape=[1] (1 variável de entrada)
    ]
)

# 3. Compilação do Modelo
# Aqui definimos como a rede aprende:
# - optimizer: O algoritmo que ajusta os pesos (Adam é eficiente e comum)
# - loss: A métrica que mede o erro (Erro Quadrático Médio para regressão)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")

# 4. Treinamento do Modelo (Fit)
# O modelo faz a leitura dos dados por várias épocas (iterações de treino)
print("Treinando o modelo...")
history = model.fit(X, y, epochs=500, verbose=0)  # verbose=0 oculta os logs por época
print("Treinamento concluído!")

# 5. Avaliação do Modelo (Fazer Predições)
# Testando com um aluno que estudou 5 horas
horas_teste = np.array([5.0])
nota_prevista = model.predict(horas_teste)

print(
    f"\nPara {horas_teste[0]} horas de estudo, a nota estimada é: {nota_prevista[0][0]:.2f}"
)