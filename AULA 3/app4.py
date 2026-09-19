import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Pets (Passeios vs Felicidade)
pets = pd.DataFrame(
    {"passeios": [1, 2, 3, 4, 5], "felicidade": [2, 4, 5, 8, 10]}
)

# Converter para arrays NumPy
X = np.array(pets["passeios"], dtype=float)
y = np.array(pets["felicidade"], dtype=float)

# 2. Modelo (1 camada, 1 neurônio para Regressão Linear)
model = Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação
# Voltamos para o 'mse' (Mean Squared Error), já que estamos prevendo um número contínuo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")

# 4. Treinamento
print("Treinando a rede neural...")
history = model.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído!")

# 5. Predição
# Prevendo a felicidade de um pet que passeia 6 vezes na semana
passeios_novo = np.array([6.0])
felicidade_prevista = model.predict(passeios_novo)

print(
    f"\nPara {passeios_novo[0]:.0f} passeios, a felicidade estimada do pet é: {felicidade_prevista[0][0]:.2f}"
)