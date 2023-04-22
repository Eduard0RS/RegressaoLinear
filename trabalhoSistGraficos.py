import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
df = pd.read_csv('Income1.csv')

# Extrair as variáveis de entrada (x) e de saída (y)
x= df['Education'].values.reshape(-1, 1)

y= df['Income'].values.reshape(-1, 1)

# Método sem adição dos pontos x 

w=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

y_pred = w * x

#Curva do erro quadrático médio

w_range = np.linspace(-100, 100, 10000)

values=[]
for w in w_range:
    y_pred2 = w * x
    values.append(np.mean((y-y_pred2)**2))

minMse = np.argmin(values)
w_minMse = w_range[minMse]


# Criar figura com subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 linha, 2 colunas

# Plotar regressão linear com adição dos pontos x no primeiro subplot
axs[0].plot(x,y_pred, color='red', label='Regressão Linear')
axs[0].scatter(x, y, label='Dados')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].legend()
axs[0].set_title('Regressão Linear sem Adição dos Pontos X')

axs[1].plot(w_range, values, color='red', label='Erro Quadrático Médio')
axs[1].scatter(w_minMse, values[minMse], color='green', label='Mínimo do Erro Quadrático Médio')
axs[1].annotate('Mínimo do Erro Quadrático Médio: '+ str(round(w_minMse, 2)), xy=(w_minMse, values[minMse]))
axs[1].set_xlabel('W')
axs[1].set_ylabel('Erro Quadrático Médio')
axs[1].legend()
axs[1].set_title('Curva do Erro Quadrático Médio')

# Exibir os subplots
plt.show()









