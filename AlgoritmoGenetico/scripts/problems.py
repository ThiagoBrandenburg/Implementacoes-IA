import numpy as np
import math

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
    

class AlgebricFunction:
    def __init__(self,config:dict,precision=0.001) -> None:
        self.x_min, self.x_max = self.parse_range(config)
        self.precision = precision
        self.y_max = self.set_y_max(self.x_max)
        self.y_min = self.set_y_min(self.x_min)
        self.limit = self.set_bit_limit()

    def parse_range(self,config):
        bound = list(map(int,config['BOUND'].strip('][ ').split(',')))
        return bound
    
    def set_bit_limit(self):
        print(self.x_max,self.x_min,self.precision)
        lim = (self.x_max - self.x_min)/self.precision
        print('set_bit_limit',lim)
        l = 0
        while math.pow(2,l) < lim: l+=1
        return l

    def set_problem(self,config:dict)->dict:
        config['COD'] = 'BIN'
        config['DIM'] = str(self.limit)
        return config

    def encode(self,solution:float):
        d = round((solution - self.x_min) / ((self.x_max-self.x_min)/(math.pow(2,self.limit))-1))
        cromossomo = []
        for _ in range(self.limit):
            cromossomo.append(d%2)
            d = d//2
        cromossomo
        return cromossomo
            
    def decode(self,cromossomo:list)->list:
        d = sum([cromossomo[i]*math.pow(2,i) for i in range(self.limit)])
        solution = self.x_min + ((self.x_max - self.x_min)/(math.pow(2,self.limit)-1))*d
        return solution


    def objective_function(self,solution)->any:
        x  = solution[0]
        fx = math.cos(20*x) - (abs(x)/2) + (math.pow(x,3)/4)
        return fx
    
    def set_y_max(self,x):
        max = math.pow(x,3)/4 + 1
        return max

    def set_y_min(self,x):
        min = math.pow(x,3)/4 - abs(x)/2 -1
        return min

    def fitness(self,solution):
        fit_value = (self.objective_function(solution) - self.y_min)/(self.y_max)-(self.y_min)
        return fit_value

    def fit_max(self,solution:list):
        return self.fitness(solution)

    def fit_min(self,solution:list):
        return 1 - self.fitness(solution)

                    
                