import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Café (Xícaras vs Nível de Energia)
cafe = pd.DataFrame(
    {"xicaras": [1, 2, 3, 4, 5], "energia": [2, 4, 6, 8, 10]}
)

X = np.array(cafe["xicaras"], dtype=float)
y = np.array(cafe["energia"], dtype=float)

# 2. Construção da Rede Neural (1 camada, 1 neurônio)
model = Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação do Modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")

# 4. Treinamento
print("Treinando o modelo...")
history = model.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído!")

# 5. Predição
# Prevendo o nível de energia ao tomar 6 xícaras de café
xicaras_teste = np.array([6.0])
energia_prevista = model.predict(xicaras_teste)

print(
    f"\nAo tomar {xicaras_teste[0]:.0f} xícaras de café, o nível de energia estimado é: {energia_prevista[0][0]:.2f}"
)