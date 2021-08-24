__author__ = "Alexander Michel"
__email__ = "michelal@students.uni-marburg.de"

from time import perf_counter
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool

import numpy as np

import random
import time


class BinaryMVO:
    """
    A BinaryMVO object is an instance of a binary multi verse optimization

    :param n_universes: number of universes in the multi verse
    :type n_universes: int
    :param d: dimension of the solution to the optimization problem
    :type d: int
    :param f: fitness function to evaluate the fitness of a solution
    :param f_args: all arguments that have to get parsed to the fitness function (except solution)
    :type f_args: dict
    :param p: The probabilities associated with each entry in d.
    :type p: 1-D array_like, optional
    :param funker_name: name of docker swarm service that is responsible for evaluating the fitness
    :type funker_name: str
    :param log_path: path where the log file gets saved
    :type log_path: str
    :param log_file_name: file name of the log file
    :type log_file_name: str
    :param new_random_state_each_generation: set if a new random state should be created for every generation so that
                                            for every fitness evaluation in one generation there is no difference
                                            because of randomness
    :type new_random_state_each_generation: bool
    :param n_jobs: Parallize initialization. Uses all cores if set to -1. Sequential initialization, if n_jobs == 0.
                                             Uses n_jobs otherwise.
    :type n_jobs: int
    """

    def __init__(self, n_universes, d, f=None, f_args=None, p=None, funker_name=None, log_path=None, log_file_name=None,
                 new_random_state_each_generation=False, n_jobs=-1, initialize_with=None):
        print("INITIALIZING " + str(n_universes) + " UNIVERSES...")
        start_perf = perf_counter()
        self.p = p
        self.universes = []
        self.new_random_state_each_gen = new_random_state_each_generation
        if self.new_random_state_each_gen:
            rs = random.randint(1, 100000)
        else:
            rs = None
        if n_jobs == 0:
            for i in range(n_universes):
                self.universes.append(BinaryUniverse(d=d, f=f, f_args=f_args, p=self.p,
                                                     funker_name=funker_name, random_state=rs,
                                                     initialize_with=initialize_with))
                print("Appended one universe...")
        else:
            p = ProcessingPool(n_jobs)
            universes = p.map(
                lambda _: BinaryUniverse(d=d, f=f, f_args=f_args, p=self.p, funker_name=funker_name, random_state=rs,
                                         initialize_with=initialize_with),
                range(n_universes)
            )
            self.universes += universes
        self.funker_name = funker_name
        self.log_path = log_path
        self.log_file_name = log_file_name
        self.global_best_solution = []
        self.global_best_fitness = np.inf
        self.mean_fitness = np.inf
        self.global_best_var_metrics = None
        self.dimension = d
        self.n_universes = n_universes
        self.sorted_universes = self.universes.copy()
        self._check_new_best()
        self._sort_universes()
        self._set_normalized_fitness()
        self._calc_mean_fitness()
        self._print_log_message(init=True, start_perf=start_perf)
        self.WEP = 0  # wormhole existance probability
        self.TDR = 0  # travelling distance rate
        self.gen = 1

    def run(self, threshold, max_iterations=200, p=6.0, min_wep=0.2, max_wep=1, parallel=False):
        """
        run the optimization until a threshold for the fitness value or a number of generations is reached
        :param threshold: fitness value threshold
        :type threshold: float
        :param max_iterations: maximum number of generations
        :type max_iterations: int
        :param p: parameter which is used to calculate the travelling distance rate
        :type p: float
        :param min_wep: minimum value for whormhole existance probability
        :type min_wep: float
        :param max_wep: maximum value for whormhole existance probability
        :type max_wep: float
        :param parallel: set if algorithm should run in parallel
        :type parallel: bool
        :return: the best solution found by the optimizer and the best fitness value
        :rtype: list, float
        """
        if parallel:
            if self.funker_name is None:
                pool = Pool()
            else:
                pool = Pool(self.n_universes)
        while self.global_best_fitness > threshold and self.gen <= max_iterations:
            start_perf = perf_counter()
            self._set_normalized_fitness()
            self._sort_universes()
            self.WEP = min_wep + self.gen * ((max_wep - min_wep) / max_iterations)
            self.TDR = 1 - (self.gen / max_iterations) ** (1 / p)
            for i in range(1, self.n_universes):
                black_hole_index = i
                for j in range(self.dimension):
                    r1 = random.random()
                    if r1 < self.universes[i].normalized_fitness:
                        white_hole_index = self._roulette_selection()
                        self.universes[black_hole_index].solution[j] = self.sorted_universes[white_hole_index].solution[
                            j]
                    r2 = random.random()
                    if r2 < self.WEP:
                        r3 = random.random()
                        r4 = random.random()
                        M = abs(self.global_best_solution[j] - self.universes[i].solution[j]) + self.TDR * r4
                        r5 = random.random()
                        if r5 < _vshaped(M):
                            flip = np.random.binomial(1, self.p[0])
                            if flip == 1:
                                self.universes[i].flip_bit(j)
            if self.new_random_state_each_gen:
                rs = random.randint(1, 1000000)
                for u in self.universes:
                    u.random_state = rs
            evaluated_universes = []
            if parallel:
                if self.funker_name is None:
                    evaluated_universes = pool.map(_eval_fit, self.universes)
                    self.universes = evaluated_universes.copy()
                else:
                    map_result = pool.map_async(_eval_fit_docker, self.universes, callback=evaluated_universes.append)
                    map_result.wait()
                    self.universes = evaluated_universes.copy()[0]
            else:
                for u in self.universes:
                    evaluated_universes.append(_eval_fit(u))
                self.universes = evaluated_universes.copy()
            self._check_new_best()
            self._calc_mean_fitness()
            self._print_log_message(init=False, start_perf=start_perf)
            self.gen += 1
        return self.global_best_solution, self.global_best_fitness

    def _check_new_best(self):
        """
        checks if there is an ant with better fitness than the global best fitness
        """
        for u in self.universes:
            if u.best_fitness < self.global_best_fitness:
                self.global_best_fitness = u.best_fitness
                self.global_best_var_metrics = u.best_var_metrics
                self.global_best_solution = u.best_solution.copy()

    def _sort_universes(self):
        """
        sort universes based on fitness value
        """
        self.universes = sorted(self.universes, key=lambda universe: universe.fitness)
        self.sorted_universes = self.universes.copy()
        """
        self.sorted_universes = sorted(self.universes, key=lambda universe: universe.fitness)
        """

    def _set_normalized_fitness(self):
        """
        set the normalized fitness value for each universe
        """
        fitness_list = []
        for u in self.universes:
            fitness_list.append(u.fitness)
        min_fitness = min(fitness_list)
        max_fitness = max(fitness_list)
        if min_fitness == max_fitness:
            for u in self.universes:
                u.normalized_fitness = 0
        else:
            for u in self.universes:
                u.normalized_fitness = (u.fitness - min_fitness) / (max_fitness - min_fitness)
        """
        sum_of_fitness = 0
        for u in self.universes:
            sum_of_fitness += u.fitness**2
        sum_of_fitness = sum_of_fitness**0.5
        for u in self.universes:
            u.normalized_fitness = u.fitness/sum_of_fitness
        """

    def _roulette_selection(self):
        """
        roulette selection to select one universe based on the fitness

        !!!UNIVERSES HAVE TO BE SORTED BEFORE USE!!!
        """
        fitness_list = []
        for u in self.sorted_universes:
            fitness_list.append(u.fitness)
        fitness_list = [1 / n for n in fitness_list]
        fitness_list.reverse()
        sum_fitness_list = 0
        for i in fitness_list:
            sum_fitness_list += i
        normalized_fitness_list = []
        for x in fitness_list:
            normalized_fitness_list.append(x / sum_fitness_list)
        summed_probs = list(np.cumsum(normalized_fitness_list))
        r = random.random() * summed_probs[-1]
        for i in range(self.n_universes):
            if r < summed_probs[i]:
                return self.n_universes - i - 1
        return self.n_universes - 1
        """

        fitness_list = []
        for u in self.sorted_universes:
            fitness_list.append(u.fitness)
        fitness_list = [-n for n in fitness_list]
        normalized_fitness_list = fitness_list
        summed_probs = list(np.cumsum(normalized_fitness_list))
        r = random.random()*summed_probs[-1]
        res = 0
        for i in range(self.n_universes):
            if summed_probs[i] > r:
                res = i
                return res
        return res
        """

    def _print_log_message(self, init, start_perf):
        """
        print (and save) a log message to check progress in every generation
        :param init: if true the initialization log message gets printed
        :type init: bool
        :param start_perf:  timer that was set at the beginning of the current generation to check elapsed time in this
                            generation
        """
        log_msg = 50 * "-"
        if init:
            log_msg += "\nINITIALIZATION SUCCESSFUL"
        else:
            log_msg += "\nGENERATION: " + str(self.gen)
        log_msg += "\nBest Fitness: " + str(self.global_best_fitness)
        log_msg += ", best metrics: " + str(self.global_best_var_metrics)
        log_msg += "\nCurrent generation mean fitness: " + str(self.mean_fitness)
        log_msg += "\nBest solution: " + str(list(np.nonzero(self.global_best_solution)[0]))
        log_msg += "\nTime elapsed: " + str(perf_counter() - start_perf) + " seconds\n"
        print(log_msg)
        if self.log_path is not None and self.log_file_name is not None:
            with open(self.log_path + self.log_file_name, 'a', encoding="utf-8") as file:
                file.write(log_msg)

    def _calc_mean_fitness(self):
        """
        calculates the mean fitness of the multi verse
        """
        tmp = 0
        for u in self.universes:
            tmp += u.fitness
        tmp = tmp / self.n_universes
        self.mean_fitness = tmp


def _vshaped(M):
    """
    calculation of a vshaped function to calculate wether a bit gets flipped or not
    """
    return abs((2 / np.pi) * np.arctan((np.pi / 2) * M))


class BinaryUniverse:
    """
    A BinaryUniverse instance is a object representing a universe in Binary Multi Verse Optimization
    :param d: dimension of the solution
    :type d: int
    :param f: fitness function
    :param f_args: all arguments that have to get parsed to the fitness function (except solution)
    :type f_args: dict
    :param p: The probabilities associated with each entry in d.
    :type p: 1-D array_like, optional
    :param funker_name: name of docker swarm service that is responsible for evaluating the fitness
    :type funker_name: str
    :param random_state: set if a new random state should be created for every generation so that
                        for every fitness evaluation in one generation there is no difference
                        because of randomness
    :type random_state: Optional[int]
    """

    def __init__(self, d, f=None, f_args=None, p=None, funker_name=None, random_state=None, initialize_with=None):
        rng = np.random.default_rng()
        if initialize_with is None:
            self.solution = rng.choice([1, 0], size=d, p=[0.5, 0.5] if p is None else p).tolist()
        else:
            # init with custom vector, however slightly randomize each individual
            self.solution = initialize_with
            self.randomize_custom_individual()
        self.best_solution = self.solution.copy()
        self.fitness = np.inf
        self.normalized_fitness = 0
        self.best_fitness = np.inf
        self.best_var_metrics = None
        self.random_state = random_state
        self.f_args = f_args
        self.fitness_function = f
        self.funker_name = funker_name
        self.eval_fitness()

    def randomize_custom_individual(self):
        rng = np.random.default_rng()
        d = len(self.solution)
        idcs = np.nonzero(self.solution)[0]
        flip_ones = \
            idcs[list(set(rng.integers(low=0, high=len(idcs), size=int(len(idcs) / 2))))]
        flip_zeros = \
            np.arange(d)[list(set(rng.integers(low=0, high=d, size=int(len(idcs) / 2))))]
        for j in np.append(flip_ones, flip_zeros):
            self.flip_bit(j)

    def flip_bit(self, j):
        """
        flips bit j in solution vector
        :param j: number of bit which is to be flipped (min: 0, max: d - 1)
        :type j: int
        """
        if self.solution[j] == 1:
            self.solution[j] = 0
        elif self.solution[j] == 0:
            self.solution[j] = 1
        else:
            raise ValueError("part of the solution is not 0 or 1")

    def eval_fitness(self):
        """
        evaluates the fitness of the universe
        """
        self.fitness, var_metrics = \
            self.fitness_function(self.solution, random_state=self.random_state, **self.f_args)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_solution = self.solution.copy()
            self.best_var_metrics = var_metrics

    def eval_fitness_docker(self):
        """
        evaluates the fitness of the universe via docker swarm
        """
        import funker
        myargs = self.f_args.copy()
        myargs["position"] = self.solution
        myargs["random_state"] = self.random_state
        failed = True
        while failed:
            try:
                self.fitness, acc, div, f1 = funker.call(self.funker_name, **myargs)
                failed = False
            except:
                time.sleep(0.1)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_solution = self.solution.copy()
            self.best_acc = acc
            self.best_div = div
            self.best_f1 = f1


def _eval_fit(u: BinaryUniverse):
    """
    evaluate the fitness of an universe
    :param u: one universe out of the multi verse
    :type u: BinaryUniverse
    :return: universe with updated fitness
    """
    u.eval_fitness()
    return u


def _eval_fit_docker(u: BinaryUniverse):
    """
    evaluate the fitness of an universe via docker swarm
    :param u: one ant out of the multi verse
    :type u: BinaryUniverse
    :return: universe with updated fitness
    """
    u.eval_fitness_docker()
    return u
