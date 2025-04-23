import math
import random
import numpy as np

class RBM():
    def __init__(self, nv, nh, epsilon=0.01, training_stop_diff=0.1):
        self.a = np.random.uniform(low=-1, high=1, size=nv)
        self.b = np.random.uniform(low=-1, high=1, size=nh)
        self.w = np.random.uniform(low=0, high=1, size=(nv, nh))
        self.epsilon = epsilon
        self.training_stop_diff = training_stop_diff

    def train(self, training_set, max_iterations):
        for _ in range(max_iterations):
            if self.training_step(training_set):
                return

    def training_step(self, training_set):
        is_done = False

        v0 = random.choice(training_set)
        v1, h1, Pv, Ph = self.calc_next_v(v0)

        old_a, old_b, old_w = np.copy(self.a), np.copy(self.b), np.copy(self.w)
        self.update_params(v0, v1, Ph, h1)
        if RBM.calc_diff(old_a, self.a) < self.epsilon * self.training_stop_diff and \
           RBM.calc_diff(old_b, self.b) < self.epsilon * self.training_stop_diff and \
           RBM.calc_diff(old_w.flatten(), self.w.flatten()) < self.epsilon * self.training_stop_diff:
            is_done = True

        return is_done

    def calc_next_v(self, v0, T=1):
        Eh = ((v0 @ self.w) + self.b)
        Ph = RBM.sigma(Eh / T)
        h1 = np.array([1 if random.random() <= Ph[i] else 0 for i in range(len(Ph))])

        Ev = ((h1 @ np.transpose(self.w)) + self.a)
        Pv = RBM.sigma(Ev / T)
        v1 = np.array([1 if random.random() <= Pv[i] else 0 for i in range(len(Pv))])
        return v1, h1, Pv, Ph

    @staticmethod
    def sigma(x):
        return 1.0 / (1.0 + math.e ** -x)

    def update_params(self, v0, v1, Ph, h1):
        self.a += self.epsilon * (v0 - v1)
        self.b += self.epsilon * (Ph - h1)
        self.w += self.epsilon * (np.outer(v0, Ph) - np.outer(v1, h1))

    @staticmethod
    def calc_diff(curr, prev):
        diff = 0.0
        for i in range(len(curr)):
            diff += abs(curr[i] - prev[i])
        return diff / len(curr)

    def infer(self, v0, in_mask, initial_temp, stopping_energy_diff=0.1):
        T = initial_temp
        # randomise output neurons
        curr_v0 = np.array([v0[i] if in_mask[i] else random.randint(0, 1) for i in range(len(v0))])
        _, h0, _, _ = self.calc_next_v(curr_v0, T)
        prev_energy = self.calc_energy(curr_v0, h0)
        niterations_without_energy_change = 0

        while True:
            v1, h1, Pv, _ = self.calc_next_v(curr_v0, T)

            curr_v0 = np.array([v0[i] if in_mask[i] else v1[i] for i in range(len(v0))])
            curr_energy = self.calc_energy(curr_v0, h1)

            if abs(prev_energy - curr_energy) < stopping_energy_diff:
                niterations_without_energy_change += 1
            else:
                niterations_without_energy_change = 0

            if niterations_without_energy_change == 5:
                return curr_v0

            prev_energy = curr_energy
            T = 0.8 * T

    # calculating the total energy of the network according to the formula in page 123
    def calc_energy(self, v, h):
        weights_sum = 0
        for i in range(len(v)):
            for j in range(len(h)):
                weights_sum += self.w[i][j] * v[i] * h[j]
        return -np.dot(self.a, v) - np.dot(self.b, h) - weights_sum
