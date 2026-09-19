import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 1. Dataset de Pizzas (Tamanho em cm vs Preço em R$)
pizza = pd.DataFrame(
    {
        "tamanho": [20, 25, 30, 35, 40],
        "preco": [20, 30, 40, 50, 60],
    }
)

X = np.array(pizza["tamanho"], dtype=float)
y = np.array(pizza["preco"], dtype=float)

# 2. Construção da Rede Neural (1 camada, 1 neurônio)
model = Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação do Modelo
# Usamos o otimizador Adam e a função de perda MSE (Erro Quadrático Médio)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mse"
)

# 4. Treinamento
print("Treinando o modelo de preços de pizza...")
history = model.fit(X, y, epochs=1000, verbose=0)
print("Treinamento concluído!")

# 5. Predição
# Estimando o preço de uma pizza de 32 cm
tamanho_teste = np.array([32.0])
preco_previsto = model.predict(tamanho_teste)

print(
    f"\nPara uma pizza de {tamanho_teste[0]:.0f} cm, o preço estimado é: R$ {preco_previsto[0][0]:.2f}"
)