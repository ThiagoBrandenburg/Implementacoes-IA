from scripts.leitor import Leitor
from scripts.ambiente import Ambiente
from scripts.problems import Nrainhas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.cm as cmap

leitor = Leitor()
config = leitor.loadConfig("AlgoritmoGenetico/data/data_nrainhas8.txt")
nrainhas = Nrainhas(int(config['DIM']))


#Carregando problema
ambiente = Ambiente(config=config,problem=nrainhas)
# print('config:',ambiente.config)
# print('pop:',ambiente.population)
# print('instance:',[nrainhas.decode(cromossomo) for cromossomo in ambiente.population])

#Definindo melhor e pior solução
ambiente.evaluate()
best = ambiente.population[ambiente.population.index(max(ambiente.population))]
worst = ambiente.population[ambiente.population.index(min(ambiente.population))]

fig, ax = plt.subplots(1,2,figsize=(10,4))
sns.heatmap(ax=ax[0],data=nrainhas.get_matrix(nrainhas.decode(best)),cmap='gray').set(title='Melhor')
sns.heatmap(ax=ax[1],data=nrainhas.get_matrix(nrainhas.decode(worst)),cmap='gray').set(title='Pior')
plt.show()

print(nrainhas.decode(best))
# print('best:\n',nrainhas.get_matrix(nrainhas.decode(best)))
# print('worst:\n',nrainhas.get_matrix(nrainhas.decode(worst)))

