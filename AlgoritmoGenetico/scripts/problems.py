import numpy as np
import math
import random
import pandas as pd
from enum import Enum


class Problem:
    def encode(self, solution) -> np.array:
        ...

    def decode(self, cromossomo: np.array) -> any:
        ...

    def objective_function(self, solution) -> any:
        ...

    def penality_function(self, solution) -> any:
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
    max_colision: int
    resolution: int

    def __init__(self) -> None:
        self.max_colision = None
        self.resolution = None

    def set_problem(self, config: dict) -> dict:
        """
        Comunicação entre o problema e o algoritmo.
        O ambiente passa uma configuração para o problema,
        que seta as configurações, e as retorna
        para o ambiente com parametros adicionais (caso necessário)
        """
        dim = int(config["DIM"])
        config["BOUND"] = (
            config["BOUND"] if "BOUND" in config.keys() else "[(0," + str(dim) + ")]"
        )
        self.penality = (
            float(config["PENALITY"]) if "PENALITY" in config.keys() else -1.0
        )
        self.resolution = int(config["DIM"])
        config["COD"] = "INT-PERM"
        self.max_colision = self._sumpa(1, self.resolution, self.resolution)
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
        fit_value = (
            self.max_colision - self.objective_function(solution)
        ) / self.max_colision
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
        fit_value = (self.objective_function(solution) / 1360) + (
            self.penality_factor * self.penality_function(solution)
        )
        return fit_value

    def set_problem(self, config: dict) -> dict:
        config["COD"] = "INT"
        # config["DIM"] = 24
        # config["BOUND"] = [0, 3]
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


class NrainhasSum:
    max_colision: int
    max_fit_value: float

    def __init__(self) -> None:
        self.penality = -1
        self.resolution = None
        self.max_colision = None
        self.max_fit_value = None

    def set_problem(self, config: dict) -> dict:
        """
        Comunicação entre o problema e o algoritmo.
        O ambiente passa uma configuração para o problema,
        que seta as configurações, e as retorna
        para o ambiente com parametros adicionais (caso necessário)
        """
        dim = int(config["DIM"])
        config["BOUND"] = (
            config["BOUND"] if "BOUND" in config.keys() else "[(0," + str(dim) + ")]"
        )
        self.penality = (
            float(config["PENALITY"]) if "PENALITY" in config.keys() else -1.0
        )
        self.resolution = int(config["DIM"])
        config["COD"] = "INT-PERM"
        self.max_colision = self._sumpa(1, self.resolution, self.resolution)
        self.max_fit_value = self.objective_function(
            [(i, self.resolution) for i in range(self.resolution)]
        )
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

    def _number_of_colisions(self, solution):
        """Atualmente utilizando só as diagonais"""
        colisions = sum(
            [
                self._diagonal(solution[i], solution[j])
                for i in range(self.resolution)
                for j in range(i)
            ]
        )
        return colisions

    def penality_function(self, solution):
        penality_value = self._number_of_colisions(solution) / self.max_colision
        return penality_value

    def objective_function(self, solucao: list[tuple[int, int]]):
        value = 0.0
        for coor in solucao:
            linha = coor[0] + 1
            coluna = coor[1] + 1
            k = coor[0] * self.resolution + coluna
            # print('X:',coor[0],'y:',coor[1],'k',k)
            value += math.sqrt(k) if linha % 2 == 1 else math.log(k, 10)
        return value

    def is_valid(self, solution):
        if self._number_of_colisions(solution) == 0.0:
            return True
        else:
            False

    def fitness(self, solution):
        """Perfect solution is one, worst solution is zero"""
        # print("max fit value:", self.max_fit_value)

        part1 = self.objective_function(solution) / self.max_fit_value
        # print('part1',part1, 'fit(solution)=',self.objective_function(solution),'max_fit_value=',self.max_fit_value)

        part2 = self.penality * self.penality_function(solution)

        fit_value = part1 + part2
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


class Labirinto:
    """
    Codificação é um vetor de inteiros de 100 posições
    os movimentos são parado, direita, esquerda, cima, baixo
    """

    lab_map: np.ndarray
    lab_resolution: tuple[int, int]
    path_size: int
    start: tuple[int, int]
    end: tuple[int, int]

    class Tile(Enum):
        WALL = 0
        PATH = 1
        START = 2
        END = 3

    class Move(Enum):
        STAND = 0
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

    def __init__(self) -> None:
        self.lab_map = None
        self.lab_resolution = None
        self.path_size = None

    def set_problem(self, config: dict) -> dict:
        self.lab_map = np.array(config["MAP"])
        self.lab_resolution = self.lab_map.shape
        self.max_distance = self._euclidian_distance((0, 0), self.lab_resolution)
        self.path_size = config["DIM"]
        self.penality = (
            float(config["PENALITY"]) if "PENALITY" in config.keys() else -1.0
        )
        aux = np.where(self.lab_map == self.Tile.START.value)
        self.start = (aux[0][0], aux[1][0])
        aux = np.where(self.lab_map == self.Tile.END.value)
        self.end = (aux[0][0], aux[1][0])
        config["BOUND"] = (
            "[" + ",".join(["(0.0,0.9999999)" for _ in range(self.path_size)]) + "]"
        )
        return config

    def _euclidian_distance(self, p1: tuple, p2: tuple):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def next_tile(self, tile1, move):
        i, j = tile1
        match move:
            case self.Move.UP.value:
                i -= 1
            case self.Move.DOWN.value:
                i += 1
            case self.Move.LEFT.value:
                j -= 1
            case self.Move.RIGHT.value:
                j += 1
        return (i, j)

    def possible_moves(self, tile, history):
        moves = []
        for move in self.Move:
            next = self.next_tile(tile, move.value)
            tile_value = self.lab_map[next[0]][next[1]]
            if (
                (tile_value != self.Tile.WALL.value)
                and (next[0] > 0)
                and (next[1] > 0)
                and next not in history
            ):
                moves.append(move.value)
        if len(moves) == 0:
            return [self.Move.STAND.value]
        else:
            return moves

    def decode(self, cromossomo: np.array) -> any:
        current_tile = self.start
        solution = [current_tile]
        for alelo in cromossomo:
            possibilites = self.possible_moves(current_tile, solution)
            # print('len(possibilites)',len(possibilites),' alelo:',alelo)
            chosen_pos = math.floor(len(possibilites) * alelo)
            chosen_move = possibilites[chosen_pos]
            next_t = self.next_tile(current_tile, chosen_move)
            solution.append(next_t)
            current_tile = next_t
        return solution

    def encode(self, solution) -> np.array:
        pass

    def _encode_move(self, tile1, tile2):
        if tile1[0] > tile2[0]:
            return self.Move.UP.value
        elif tile1[0] < tile2[0]:
            return self.Move.DOWN.value
        elif tile1[1] > tile2[1]:
            return self.Move.LEFT.value
        elif tile1[1] < tile2[1]:
            return self.Move.RIGHT.value
        else:
            return self.Move.STAND

    def objective_function(self, solution) -> any:
        value = self._euclidian_distance(solution[-1], self.end)
        return value

    def _path_size(self, solution) -> any:
        value = sum(
            [0.0 if move == self.Move.STAND.value else 1.0 for move in solution]
        )
        return value

    def penality_function(self, solution) -> any:
        value = self._path_size(solution) / self.path_size
        return value

    def fitness(self, solution) -> any:
        fit_value = 1.0 - (self.objective_function(solution) / self.max_distance)
        penality_value = self.penality_function(solution)
        value = fit_value + self.penality * penality_value
        return value

    def fit_max(self, solution) -> any:
        return self.fitness(solution)

    def fit_min(self, solution) -> any:
        return 1 - self.fitness(solution)

    def get_matrix(self, solution):
        best_map = np.array(
            [
                [1 if (i, j) in solution else 0 for j in range(self.lab_map.shape[1])]
                for i in range(self.lab_map.shape[0])
            ]
        )
        total_map = best_map + self.lab_map
        return total_map

    def generate_population(self, pop_size) -> list:
        pass


class MetroOld:
    def __init__(self) -> None:
        pass

    def set_dataframe(self, config, index_col, sheet_name):
        df = pd.read_excel(config["MAP"], index_col=index_col, sheet_name=sheet_name)
        df.fillna(0.0, inplace=True)
        df = df + df.transpose()
        df.replace(0.0, np.inf, inplace=True)
        for el in df.columns:
            df.loc[el, el] = 0.0
        return df

    def set_problem(self, config: dict) -> dict:
        print("UEPA")
        self.distance_dataframe = self.set_dataframe(
            config=config, index_col="Estacao", sheet_name="distancia_real"
        )
        self.straight_distance_dataframe = self.set_dataframe(
            config, index_col="Estacao", sheet_name="distancia_reta"
        )
        # self.distance_dataframe = pd.read_excel(config['MAP'],
        #                                        index_col='Estacao',
        #                                        sheet_name='distancia_real')
        #
        # self.straight_distance_dataframe = pd.read_excel(config['MAP'],
        #                                                 sheet_name='distancia_reta',
        #                                                 index_col='Estacao')
        self.number_of_stations = self.distance_dataframe.columns.__len__() - 1
        self.stations = list(self.distance_dataframe.columns[1:])
        self.start = config["START"]
        self.end = config["END"]
        self.velocity = config["VELOCIDADE"]
        self.stop_time = 5 / 60
        print(self.stations, self.start, self.end)
        config["DIM"] = self.number_of_stations - 1
        self.penality = (
            float(config["PENALITY"]) if "PENALITY" in config.keys() else -1.0
        )
        config["COD"] = "INT-PERM"
        config["BOUND"] = "[(-1," + str(self.number_of_stations - 2) + ")]"
        # config['BOUND'] = '['+','.join(['(0.0,0.9999999)' for _ in range(self.path_size)])+']'
        possible_paths = self.stations.copy()
        possible_paths.remove(self.start)
        possible_paths.remove(self.end)
        zipped_path = [(i, j) for i, j in enumerate(possible_paths)]
        self.stationDecoderDict = dict(zipped_path)
        return config

    def _distance(self, e1, e2):
        """return distance between stations"""
        value = self.distance_dataframe.loc[e1, e2]
        return value

    def _straight_distance(self, e1, e2):
        """return distance between stations"""
        value = self.straight_distance_dataframe.loc[e2, e1]
        return value

    def encode(self, solution) -> np.array:
        ...

    def decode(self, cromossomo: np.array) -> any:
        solution = [self.start]
        for alelo in cromossomo:
            if alelo == -1:
                break
            else:
                solution.append(self.stationDecoderDict[alelo])
        solution.append(self.end)
        return solution

    def objective_function(self, solution) -> any:
        """Retorna o tempo de viagem em horas"""
        tempo_de_parada = (len(solution) - 2) * self.stop_time
        tempo_de_trajeto = 0.0
        pivot = self.start
        for estacao in solution[1:-1]:
            value = self._distance(pivot, estacao)
            if np.isnan(value):
                value = self._straight_distance(pivot, estacao)
            tempo_de_trajeto += value
            pivot = estacao
        tempo_de_trajeto /= self.velocity
        tempo_total = tempo_de_parada + tempo_de_trajeto
        return tempo_total

    def penality_function(self, solution) -> any:
        value = sum(
            [
                1.0 if np.isnan(self._distance(solution[i], solution[i + 1])) else 0.0
                for i in range(len(solution) - 1)
            ]
        )/self.number_of_stations
        return value

    def fitness(self, solution) -> any:
        value = (1-self.objective_function(solution))/8 + self.penality*self.penality_function(solution)
        return value

    def generate_population(self, pop_size) -> list:
        ...

    def fit_max(self, solution) -> any:
        ...

    def fit_min(self, solution) -> any:
        ...


class MetroNew:
    def __init__(self) -> None:
        pass

    def set_dataframe(self, config, index_col, sheet_name,zero_diagonal=True):
        df = pd.read_excel(config["MAP"], index_col=index_col, sheet_name=sheet_name)
        df.fillna(0.0, inplace=True)
        df = df + df.transpose()
        df.replace(0.0, np.nan, inplace=True)
        if zero_diagonal is True:
            for el in df.columns:
                df.loc[el, el] = 0.0
        return df
    
    def _set_df(self,df:pd.DataFrame):
        df_trans = df.transpose()
        df = df.where(~pd.isna(df),df_trans)
        return df


    def set_problem(self, config: dict) -> dict:
        self.distance_dataframe = self.set_dataframe(
            config=config, index_col="Estacao", sheet_name="distancia_real"
        )
        self.straight_distance_dataframe = self.set_dataframe(
            config, index_col="Estacao", sheet_name="distancia_reta"
        )
        self.lines_dataframe = pd.read_excel(
            config['MAP'], index_col="Estacao", sheet_name='linhas'
            )
        self.lines_dataframe = self._set_df(self.lines_dataframe)
        self.max_distance = self.distance_dataframe.max().max()
        self.number_of_stations = self.distance_dataframe.columns.__len__() - 1
        self.path_size = self.number_of_stations
        self.stations = list(self.distance_dataframe.columns)#[1:]
        self.start = config["START"]
        self.end = config["END"]
        self.velocity = config["VELOCIDADE"]
        self.stop_time = 5 / 60
        #print(self.stations, self.start, self.end)
        config["DIM"] = self.number_of_stations - 1
        self.penality = (
            float(config["PENALITY"]) if "PENALITY" in config.keys() else -1.0
        )
        config["COD"] = "REAL"
        config["BOUND"] = (
            "[" + ",".join(["(0.0,0.9999999)" for _ in range(config["DIM"])]) + "]"
        )
        self.neighboors = {station:[] for station in self.stations}
        for origin in self.stations:
            for destination in self.stations:
                distance = self._distance(origin,destination)
                if not np.isnan(distance) and origin != destination:
                    self.neighboors[origin].append(destination)
        return config

    def _distance(self, e1, e2):
        """return distance between stations"""
        value = self.distance_dataframe.loc[e1, e2]
        return value

    def _straight_distance(self, e1, e2):
        """return distance between stations"""
        value = self.straight_distance_dataframe.loc[e1,e2]
        return value

    def _line(self,e1,e2):
        value = self.lines_dataframe.loc[e1,e2]
        return value

    def encode(self, solution) -> np.array:
        ...

    def decode(self, cromossomo: np.array) -> any:
        current_station = self.start
        solution = [current_station]
        #Cria um mapa de referencia das estações que já foram passadas
        check_map = {station: False for station in self.stations}
        check_map[current_station] = True
        for alelo in cromossomo:
            neighboors = [n for n in self.neighboors[current_station] if check_map[n] is False]
            if len(neighboors) == 0:
                break
            # print('len(neighboors)',len(neighboors),' alelo:',alelo)
            chosen_pos = math.floor(len(neighboors) * alelo)
            chosen_neighboor = neighboors[chosen_pos]
            solution.append(chosen_neighboor)
            current_station = chosen_neighboor
            check_map[chosen_neighboor] = True
            if current_station == self.end:
                break
        return solution

    def objective_function(self, solution) -> any:
        value = self._distance(solution[-1],self.end)
        if np.isnan(value):
            value = self._straight_distance(solution[-1],self.end)*2
        return value


    def penality_function(self, solution) -> any:
        tempo_de_parada = 0.0
        tempo_de_trajeto = 0.0
        pivot = self.start
        pivot_line = self._line(self.start,solution[1])
        for estacao in solution[1:-1]:
            value = self._distance(pivot, estacao)
            current_line = self._line(pivot, estacao)
            if current_line != pivot_line:
                tempo_de_parada += self.stop_time
            if np.isnan(value):
                value = self._straight_distance(pivot, estacao)
            tempo_de_trajeto += value
            pivot = estacao
        tempo_de_trajeto /= self.velocity
        tempo_total = tempo_de_parada + tempo_de_trajeto
        return tempo_total

    def fitness(self, solution) -> any:
        objetive_factor = self.objective_function(solution)
        penality_factor = self.penality_function(solution)
        value = (1-objetive_factor/self.max_distance) + self.penality*penality_factor
        return value


    def generate_population(self, pop_size) -> list:
        ...

    def fit_max(self, solution) -> any:
        ...

    def fit_min(self, solution) -> any:
        ...
