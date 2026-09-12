

import streamlit as st    # interface grafica 
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression # o tipo de treinamento do modelo


st.header('PREVISÃO DE VENDAS')


dados_vendas = pd.DataFrame({


   'investimentos':[100,200,300,550,750,800],
   'faturamento':[1200,2500,3700,3900,5500,6900]


})


st.write(dados_vendas)


# treinar os dados 


X = dados_vendas[['investimentos']]
y = dados_vendas['faturamento']


model = LinearRegression().fit(X,y) # treina o modelo com os dados


investimento =  st.number_input('Digite o investimento', value = 150)


if investimento:
    if st.button('Analisar:'):
    
        previsao = model.predict([[investimento]])[0] #previsão
        st.write(f'Faturamento -  previsto R${previsao:.2f} **')# resultado


# -----------------------------------------------

import streamlit as st    # interface grafica 
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression # o tipo de treinamento do modelo


# vendas, mes
# 10000,1
# 2000,2
# 6000,3
# 70000,4
# 90000,5
# 10000,6
# 50000,7
# 90000,8

dados = pd.read_csv('vendas.csv')
df = pd.DataFrame(dados)
print(df) 


# ------------------------------------------------

X = df[['mes']]
y = df['vendas']

modelo = LinearRegression()
modelo.fit(X, y)

mes = 9
previsao = modelo.predict([[mes]])

print(f'Previsão de vendas para o mês {mes}: {previsao[0]:.2f}')


# ---------------------------------------------

st.title('Previsão de Vendas')

st.dataframe(df)

X = df[['mes']]
y = df['vendas']

mes = 9

modelo = LinearRegression()
modelo.fit(X, y)

previsao = modelo.predict([[mes]])

st.write(f'Previsão de vendas para o mês {mes}: {previsao[0]:,.2f}')

# --------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Previsão de Vendas")

# Carregar os dados
dados = pd.read_csv("vendas.csv")

# Treinar o modelo
X = dados[["mes"]]
y = dados["vendas"]

modelo = LinearRegression()
modelo.fit(X, y)

# Mês que queremos prever
mes = 9

# Fazer a previsão
dados_previsao = pd.DataFrame({"mes": [mes]})
previsao = modelo.predict(dados_previsao)

# Adicionar a previsão ao DataFrame
nova_linha = pd.DataFrame({
    "vendas": [previsao[0]],
    "mes": [mes]
})

dados = pd.concat([dados, nova_linha], ignore_index=True)

# Mostrar o DataFrame completo
st.dataframe(dados)



