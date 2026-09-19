import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Alunos (Faltas vs Aprovado/Reprovado)
alunos = pd.DataFrame(
    {"faltas": [0, 1, 2, 5, 7, 10], "resultado": [1, 1, 1, 0, 0, 0]}
)

X = np.array(alunos["faltas"], dtype=float)
y = np.array(alunos["resultado"], dtype=float)

# 2. Modelo de Regressão Logística (1 camada, 1 neurônio com ativação Sigmoide)
model = Sequential(
    [layers.Dense(units=1, input_shape=[1], activation="sigmoid")]
)

# 3. Compilação para Classificação Binária
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"],  # Acompanha a acurácia (% de acertos)
)

# 4. Treinamento
print("Treinando o modelo de classificação...")
history = model.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído!")

# 5. Predição
faltas_teste = np.array([3.0, 8.0])
probabilidades = model.predict(faltas_teste)

for i, falta in enumerate(faltas_teste):
    prob = probabilidades[i][0]
    # Se a probabilidade for >= 50% (0.5), consideramos Aprovado (1)
    status = "Aprovado (1)" if prob >= 0.5 else "Reprovado (0)"
    print(
        f"Aluno com {falta:.0f} faltas -> Chance de Aprovação: {prob*100:.1f}% | Decisão: {status}"
    )