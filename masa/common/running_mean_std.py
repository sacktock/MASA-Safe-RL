
import numpy as np

class RunningMeanStd:

    def __init__(self, eps=1e-8, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.zeros(shape, np.float64)
        self.count = eps

    def copy(self):
        new_obj = RunningMeanStd()
        new_obj.mean = self.mean.copy()
        new_obj.var = self.var.copy()
        new_obj.count = float(self.count)
        return new_obj

    def update(self, arr):
        arr = np.array(arr, np.float64)
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

