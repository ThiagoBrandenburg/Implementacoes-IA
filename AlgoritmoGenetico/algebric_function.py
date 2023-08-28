from scripts.problems import AlgebricFunction
from scripts.leitor import Leitor

leitor = Leitor()
config = leitor.loadConfig('AlgoritmoGenetico\data\data_algebric.txt')
print('printa:',config)
af = AlgebricFunction(config,precision=0.001)
numero = 1.2
print('numero:',numero)
print('encode(numero)=',af.encode(numero))
print('decode(encode(numero))',af.decode(af.encode(numero)))