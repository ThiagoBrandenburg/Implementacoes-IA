import numpy as np
import random

class Nrainhas:
    def __init__(self,resolution=8) -> None:
        self.resolution = resolution
        self.weight = self._sumpa(1,resolution,resolution)

    def set_problem(self,config:dict)->dict:
        dim = int(config['DIM'])
        config['BOUND'] = '[0,'+str(dim)+']'
        config['COD'] = 'INT-PERM'
        return config

    def encode(self,solucao:list[tuple[int]])->list[int]:
        '''y=k*i + j sendo k a resolução do tabuleiro'''
        cromossomo = [elemento[1] for elemento in solucao]
        return cromossomo

    def decode(self,cromossomo:list[int])->list[tuple[int]]:
        solucao = [(i,cromossomo[i]) for i in range(len(cromossomo))]
        return solucao

    def _diagonal(self, queen1:tuple[int], queen2:tuple[int]):
        q1 = queen1[0]-queen1[1] == queen2[0]-queen2[1]
        q2 = queen1[0]+queen1[1] == queen2[0]+queen2[1]
        return q1 or q2

    def _colision(self, queen1:tuple[int], queen2:tuple[int])->bool:
        if queen1[0] == queen2[0] or queen1[1] == queen2[1] or self._diagonal(queen1,queen2): return True
        else: return False

    def _sumpa(self,min,max,n):
        return (n/2)*(min+max)
    
    def objective_function(self,solution):
        '''Atualmente utilizando só as diagonais'''
        objective = sum([self._diagonal(solution[i],solution[j]) for i in range(self.resolution) for j in range(i)])
        return objective
    
    def fitness(self,solution):
        '''Perfect solution is one, worst solution is zero'''
        fit_value = (self.weight - self.objective_function(solution))/self.weight
        return fit_value
    
    def fit_max(self,solution):
        return self.fitness(solution)

    def fit_min(self,solution):
        return 1-self.fitness(solution)
        
                    
    def get_matrix(self,solution):
        matrix = np.array([[0 for _ in range(self.resolution)] for _ in range(self.resolution)])
        for queen in solution:
            matrix[queen[0]][queen[1]] = 1
        return matrix
    
        
                    
                