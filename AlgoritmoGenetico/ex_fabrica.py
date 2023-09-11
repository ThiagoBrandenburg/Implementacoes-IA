from scripts.leitor import Leitor
from scripts.ambiente import Ambiente
from scripts.problems import FabricaDeRadios
import seaborn as sns
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    config = Leitor().loadConfig('AlgoritmoGenetico/data/data_fabrica_de_radios.txt')
    problem = FabricaDeRadios()
    start = time.time()
    start = time.perf_counter()

    ambiente = Ambiente(config,problem=problem,parallel=False)
    ambiente.run()

    end = time.perf_counter() -start
    print('time:',end)
    best= problem.decode(ambiente.elite_population[0])
    print('Best Solution:', best)
    print('Best Solution Objective Value (profit):',problem.objective_function(best))
    #fig, ax = plt.subplots(1,2,figsize=(12,5))
    sns.lineplot(ambiente.results_best,color='Red')#,ax=ax[0])
    sns.lineplot(ambiente.results_mean,color='Blue')#,ax=ax[0])
    #sns.heatmap(problem.get_matrix(problem.decode(ambiente.elite_population[0])),linewidths=0.5,linecolor='Gray',cmap='gray',ax=ax[1])
    plt.show()


# print('best:\n',nrainhas.get_matrix(nrainhas.decode(best)))
# print('worst:\n',nrainhas.get_matrix(nrainhas.decode(worst)))

