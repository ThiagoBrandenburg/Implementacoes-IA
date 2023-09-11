import numpy as np
import math
import random
import seaborn as sns


class Problem:
    def encode(self, solution) -> np.array:
        ...

    def decode(self, cromossomo: np.array) -> any:
        ...

    def objective_function(self, solution) -> any:
        ...

    def fitness(self, solution) -> any:
        ...

    def set_problem(self, config: dict) -> dict:
        ...

    def generate_population(self, pop_size) -> list:
        ...

    def fit_max(self, solution) -> any:
        ...

    def fit_min(self, solution) -> any:
        ...



class Nrainhas:
    def __init__(self, resolution=8) -> None:
        self.resolution = resolution
        self.weight = self._sumpa(1, resolution, resolution)

    def set_problem(self, config: dict) -> dict:
        dim = int(config["DIM"])
        config['BOUND'] = (
            config['BOUND']
            if 'BOUND' in config.keys()
            else "[(0," + str(dim) + ")]"
            )
        config["COD"] = "INT-PERM"
        return config

    def encode(self, solucao: list[tuple[int]]) -> np.array:
        """y=k*i + j sendo k a resolução do tabuleiro"""
        cromossomo = np.array([elemento[1] for elemento in solucao])
        return cromossomo

    def decode(self, cromossomo: np.array) -> list[tuple[int]]:
        solucao = [(i, cromossomo[i]) for i in range(len(cromossomo))]
        return solucao

    def _diagonal(self, queen1: tuple[int], queen2: tuple[int]):
        q1 = queen1[0] - queen1[1] == queen2[0] - queen2[1]
        q2 = queen1[0] + queen1[1] == queen2[0] + queen2[1]
        return q1 or q2

    def _colision(self, queen1: tuple[int], queen2: tuple[int]) -> bool:
        if (
            queen1[0] == queen2[0]
            or queen1[1] == queen2[1]
            or self._diagonal(queen1, queen2)
        ):
            return True
        else:
            return False

    def _sumpa(self, min, max, n):
        return (n / 2) * (min + max)

    def objective_function(self, solution):
        """Atualmente utilizando só as diagonais"""
        objective = sum(
            [
                self._diagonal(solution[i], solution[j])
                for i in range(self.resolution)
                for j in range(i)
            ]
        )
        return objective

    def fitness(self, solution):
        """Perfect solution is one, worst solution is zero"""
        fit_value = (self.weight - self.objective_function(solution)) / self.weight
        return fit_value

    def fit_max(self, solution):
        return self.fitness(solution)

    def fit_min(self, solution):
        return 1 - self.fitness(solution)

    def get_matrix(self, solution):
        matrix = np.array(
            [[0 for _ in range(self.resolution)] for _ in range(self.resolution)]
        )
        for queen in solution:
            matrix[queen[0]][queen[1]] = 1
        return matrix
    


class AlgebricFunction:
    def __init__(self, config: dict, precision=0.001) -> None:
        self.x_min, self.x_max = self.parse_range(config)
        self.precision = precision
        self.y_max = self.set_y_max(self.x_max)
        self.y_min = self.set_y_min(self.x_min)
        self.limit = self.set_bit_limit()

    def parse_range(self, config):
        bound = list(map(int, config["BOUND"].strip("][ ").split(",")))
        return bound

    def set_bit_limit(self):
        lim = (self.x_max - self.x_min) / self.precision
        l = 0
        while math.pow(2, l) <= lim:
            l += 1
        return l

    def set_problem(self, config: dict) -> dict:
        config["COD"] = "BIN"
        config["DIM"] = str(self.limit)
        return config

    def encode(self, solution: float):
        # print('solution:',solution,'max:', self.x_max, ' min:',self.x_min, ' limit:',self.limit)
        d = round(
            (solution - self.x_min)
            * ((math.pow(2, self.limit)) - 1)
            / (self.x_max - self.x_min)
        )
        # print('d:',d)
        cromossomo = []
        for _ in range(self.limit):
            cromossomo.append(d % 2)
            d = d // 2
        cromossomo
        return cromossomo

    def decode(self, cromossomo: list) -> list:
        d = sum([cromossomo[i] * math.pow(2, i) for i in range(self.limit)])
        solution = (
            self.x_min + ((self.x_max - self.x_min) / (math.pow(2, self.limit) - 1)) * d
        )
        return solution

    def objective_function(self, solution) -> any:
        x = solution
        fx = math.cos(20 * x) - (abs(x) / 2) + (math.pow(x, 3) / 4)
        return fx

    def set_y_max(self, x):
        max = math.pow(x, 3) / 4 + 1
        return max

    def set_y_min(self, x):
        min = math.pow(x, 3) / 4 - abs(x) / 2 - 1
        return min

    def fitness(self, solution):
        fit_value = (self.objective_function(solution) - self.y_min) / (
            (self.y_max) - (self.y_min)
        )
        return fit_value

    def fit_max(self, solution: list):
        return self.fitness(solution)

    def fit_min(self, solution: list):
        return 1 - self.fitness(solution)


class FabricaDeRadios:
    """
    max 30x1 + 40x2
    x1 < 24
    x2 < 16
    x1 + 2*x2 < 40
    """

    def __init__(self, penality_factor=-1):
        self.penality_factor = penality_factor

    def encode(self, solution: list):
        return solution

    def decode(self, cromossomo: list):
        return cromossomo

    def objective_function(self, solution):
        """30*24+40*16=1360"""
        objective_value = 30 * solution[0] + 40 * solution[1]
        return objective_value

    def penality_function(self, solution):
        """24 + 2*16 = 24 +32 = 56"""
        penality = ((solution[0] + 2 * solution[1]) - 40) / 16
        penality = max(0, penality)
        return penality

    def fitness(self, solution):
        fit_value = (self.objective_function(solution) / 1360) + (self.penality_factor * self.penality_function(solution))
        return fit_value

    def set_problem(self, config: dict) -> dict:
        config["COD"] = "INT"
        #config["DIM"] = 24
        #config["BOUND"] = [0, 3]
        return config

    def generate_population(self, pop_size):
        population = [
            (random.randint(0, 24), random.randint(0, 16)) for _ in range(pop_size)
        ]
        return population

    def fit_max(self, solution):
        return self.fitness(solution)

    def fit_min(self, solution):
        return 1 - self.fitness(solution)
