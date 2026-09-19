import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Conjunto de Dados
sorvete = pd.DataFrame(
    {
        "temperatura": [18, 20, 24, 27, 30, 35],
        "vendas": [20, 25, 40, 55, 70, 100],
    }
)

# Converter as colunas do pandas em arrays do NumPy (float32)
X = np.array(sorvete["temperatura"], dtype=float)
y = np.array(sorvete["vendas"], dtype=float)

# 2. Construção da Rede Neural (Regressão Linear: 1 camada, 1 neurônio)
model = Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação
# Otimizador Adam com taxa de aprendizado ligeiramente ajustada para convergência rápida
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")

# 4. Treinamento
print("Treinando a rede neural...")
history = model.fit(X, y, epochs=1000, verbose=0)
print("Treinamento concluído!")

# 5. Predição
temp_teste = np.array([32.0])
vendas_previstas = model.predict(temp_teste)

print(
    f"\nPara uma temperatura de {temp_teste[0]}°C, a estimativa é de {vendas_previstas[0][0]:.1f} sorvetes vendidos."
)