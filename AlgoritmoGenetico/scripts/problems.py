import numpy as np

class Nrainhas:
    def __init__(self,resolution=8) -> None:
        self.resolution = resolution

    def set_problem(self,config:dict)->dict:
        dim = int(config['DIM'])
        config['BOUND'] = '[0,'+str(dim**2)+']'
        config['COD'] = 'INT-PERM'
        return config

    def encode(self,solucao:list[tuple[int]])->list[int]:
        '''y=k*i + j sendo k a resolução do tabuleiro'''
        return [solucao[i][0]*self.resolution+solucao[i][1] for i in range(len(solucao))]

    def decode(self,cromossomo:list[int])->list[tuple[int]]:
        return [(cromossomo[i]//self.resolution, cromossomo[i]%self.resolution) for i in range(len(cromossomo))]

    def _diagonal(self, queen1:tuple[int], queen2:tuple[int]):
        q1 = queen1[0]-queen1[1] == queen2[0]-queen2[1]
        q2 = queen1[0]+queen1[1] == queen2[0]+queen2[1]
        return q1 or q2

    def _colision(self, queen1:tuple[int], queen2:tuple[int])->bool:
        if queen1[0] == queen2[0] or queen1[1] == queen2[1] or self._diagonal(queen1,queen2): return True
        else: return False

    def _sumpa(self,min,max,n):
        return (n/2)*(min+max)
    
    def fitness(self,solution):
        '''Perfect solution is one, worst solution is zero'''
        weigth  = 1/self._sumpa(1,self.resolution,self.resolution)
        fit_value = 1
        for i in range(self.resolution):
            for j in range(i):
                queen1 = solution[i]
                queen2 = solution[j]
                if self._colision(queen1,queen2)==True:
                    fit_value -=weigth
        return fit_value
                    
    def get_matrix(self,solution):
        matrix = np.array([[0 for _ in range(self.resolution)] for _ in range(self.resolution)])
        for queen in solution:
            matrix[queen[0]][queen[1]] = 1
        return matrix
                    
                