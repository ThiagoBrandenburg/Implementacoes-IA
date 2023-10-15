from scripts.leitor import Leitor
from scripts.ambiente import Ambiente
from scripts.problems import Nrainhas
import seaborn as sns
import matplotlib.pyplot as plt

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
ambiente.save_elite()

best = nrainhas.decode(ambiente.population[ambiente.evaluation.index(max(ambiente.evaluation))])
worst = nrainhas.decode(ambiente.population[ambiente.evaluation.index(min(ambiente.evaluation))])


print('Best:',nrainhas.decode(best))
print('Elite:',ambiente.elite_population)
print('Fit_max(best):',nrainhas.fit_max(best))
print('Fit_min(best):',nrainhas.fit_min(best))
print('Fit_max(worst):',nrainhas.fit_max(worst))
print('Fit_min(worst):',nrainhas.fit_min(worst))

fig, ax = plt.subplots(1,2,figsize=(10,4))
sns.heatmap(ax=ax[0],data=nrainhas.get_matrix(best),cmap='gray').set(title='Melhor (colisões='+str(nrainhas.objective_function(best))+')')
sns.heatmap(ax=ax[1],data=nrainhas.get_matrix(worst),cmap='gray').set(title='Pior (colisões='+str(nrainhas.objective_function(worst))+')')
plt.show()


# print('best:\n',nrainhas.get_matrix(nrainhas.decode(best)))
# print('worst:\n',nrainhas.get_matrix(nrainhas.decode(worst)))

