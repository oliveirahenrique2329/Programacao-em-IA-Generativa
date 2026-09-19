import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Heróis (Força vs Status de Herói)
herois = pd.DataFrame(
    {
        "forca": [1, 2, 3, 7, 8, 10],
        "heroi": [0, 0, 0, 1, 1, 1],  # 0 = Não, 1 = Sim
    }
)

X = np.array(herois["forca"], dtype=float)
y = np.array(herois["heroi"], dtype=float)

# 2. Modelo de Regressão Logística (1 neurônio com ativação Sigmoide)
model = Sequential(
    [layers.Dense(units=1, input_shape=[1], activation="sigmoid")]
)

# 3. Compilação
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 4. Treinamento
print("Treinando o classificador de heróis...")
history = model.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído!")

# 5. Predição
# Testando com dois personagens: um com força 2 e outro com força 9
forca_teste = np.array([2.0, 9.0])
probabilidades = model.predict(forca_teste)

print("\n--- Resultados ---")
for i, f in enumerate(forca_teste):
    prob = probabilidades[i][0]
    classe = "Herói (1)" if prob >= 0.5 else "Comum (0)"
    print(
        f"Personagem com Força {f:.0f} -> Chance de ser Herói: {prob*100:.1f}% | Classificação: {classe}"
    )