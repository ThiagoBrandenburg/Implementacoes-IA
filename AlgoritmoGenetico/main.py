from scripts.leitor import Leitor
from scripts.ambiente import Ambiente


leitor = Leitor()



ambiente = Ambiente(leitor.loadConfig(r"Implementacoes-IA\AlgoritmoGenetico\data_real.txt"))
print(ambiente.population)

