import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Músicas (BPM vs Índice Viral)
musica = pd.DataFrame(
    {
        "bpm": [80, 90, 100, 120, 140],
        "viral": [1, 2, 4, 7, 10],
    }
)

X = np.array(musica["bpm"], dtype=float)
y = np.array(musica["viral"], dtype=float)

# 2. Modelo (1 camada, 1 neurônio)
model = Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação
# Ajustamos o learning_rate para 0.01 por conta dos valores de entrada (BPMs até 140)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse"
)

# 4. Treinamento
print("Treinando o modelo de viralização de músicas...")
history = model.fit(X, y, epochs=1000, verbose=0)
print("Treinamento concluído!")

# 5. Predição
# Estimando o potencial viral de uma música de 110 BPM
bpm_teste = np.array([110.0])
viral_previsto = model.predict(bpm_teste)

print(
    f"\nPara uma música de {bpm_teste[0]:.0f} BPM, o índice viral estimado é: {viral_previsto[0][0]:.2f}"
)