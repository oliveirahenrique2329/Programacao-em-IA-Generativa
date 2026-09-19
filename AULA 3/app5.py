import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Filmes (Duração vs Nota)
filmes = pd.DataFrame(
    {
        "duracao": [80, 90, 100, 110, 120],
        "nota": [4, 5, 7, 8, 9],
    }
)

X = np.array(filmes["duracao"], dtype=float)
y = np.array(filmes["nota"], dtype=float)

# 2. Modelo (1 camada, 1 neurônio)
model = Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação
# Para dados com valores de entrada maiores (ex: 80 a 120),
# uma taxa de aprendizado ligeiramente menor (0.01 ou 0.05) evita divergências durante o treino.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse"
)

# 4. Treinamento
print("Treinando o modelo...")
history = model.fit(X, y, epochs=1000, verbose=0)
print("Treinamento concluído!")

# 5. Predição
# Estimando a nota de um filme com 105 minutos de duração
duracao_teste = np.array([105.0])
nota_prevista = model.predict(duracao_teste)

print(
    f"\nPara um filme de {duracao_teste[0]:.0f} minutos, a nota estimada é: {nota_prevista[0][0]:.2f}"
)