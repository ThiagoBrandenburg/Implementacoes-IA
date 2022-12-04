import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

#Constantes
E0 = 0
E1 = 1
E2 = 2

CORRENTE = 0
TENSAO = 1

#quero ler os dados na forma dados[E0][nro_teste][corrente ou tensao]

dados = pd.read_csv('Trabalho_Final/docs/MotorUniversal_2k_5kHz_SemNorm.xlsx - TesteMotorUniversalAgoraVai2000.csv')
print(dados.info())


'''

reader = csv.reader(open("Trabalho_Final/docs/MotorUniversal_2k_5kHz_SemNorm.xlsx - TesteMotorUniversalAgoraVai2000.csv", "r"), delimiter=",")
x = list(reader)

x.pop(0)
print(len(x))
dados = np.array([[],[],[]])

for i in range(len(x)):
    for j in range(len(x[i])):
        valor = x[i][j].replace(",",".")
        if valor=='':print('pos:',i,j)
        x[i][j] = valor

print(x[1][1])



result = np.array(x).astype("float")

print(result[1][1001])
'''    