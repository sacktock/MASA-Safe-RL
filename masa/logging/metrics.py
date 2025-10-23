
import tensorflow as tf
import numpy as np
import warnings
import time 
from tqdm import tqdm

class Metrics:

    def __init__(self, prefix=""):
        self.n = 0
        self.prefix = prefix
        self.stats = None

    def update(self, values):
        v = np.asarray(values, dtype=np.float32).ravel()

        m = v.size
        if m == 0:
            return

        if self.stats is None:
            self.n = m
            self.stats = {
                'mean': np.mean(v),
                'mean_squares': np.mean(v**2),
                'max': np.max(v),
                'min': np.min(v),
                'mag': np.max(np.abs(v)),
            }
        else:
            self.n += m
            self.stats = {
                'mean': np.mean(v) * (m/self.n) + self.stats['mean'] * ((self.n - m)/self.n),
                'mean_squares': np.mean(v**2) * (m/self.n) + self.stats['mean_squares'] * ((self.n - m)/self.n),
                'max': max(np.max(v), self.stats['max']),
                'min': min(np.min(v), self.stats['min']),
                'mag': max(np.max(np.abs(v)), self.stats['mag']),
            }

    def get(self):
        stats = {
            'mean': self.stats['mean'], 
            'std': np.sqrt(np.maximum(0.0, self.stats['mean_squares'] - self.stats['mean']**2)), 
            'max': self.stats['max'], 
            'min': self.stats['min'], 
            'mag': self.stats['mag'],
        }
        if self.prefix:
            stats = {f"{self.prefix}_{k}": v for k, v in stats.items() }
        return stats

    def __add__(self, other):
        assert isinstance(other, Summary_Stats)

        assert self.prefix == other.prefix, "can't add two Metrics objects with different prefixes"

        new = Metrics(prefix=self.prefix)
        new.n = self.n + other.n

        new.stats = {
            'mean': self.stats['mean'] * (self.n/new.n) + other.stats['mean'] * (other.n/new.n),
            'mean_squares': self.stats['mean_squares'] * (self.n/new.n) + other.stats['mean_squares'] * (other.n/new.n),
            'max': max(self.stats['max'], other.stats['max']),
            'min': min(self.stats['min'], other.stats['min']),
            'mag': max(self.stats['mag'], other.stats['mag']),
        }

        return new

class Dist:

    def __init__(self, prefix="", reservoir_size=2048, rng=None):
        self.n = 0
        self.prefix = prefix
        self.reservoir_size = int(reservoir_size)
        self.res = np.empty((0,), dtype=np.float32)
        self.rng = np.random.default_rng(rng)

    def update(self, values):
        v = np.asarray(values, dtype=np.float32).ravel()
        m = v.size
        if m == 0:
            return

        # sanity check
        assert len(v.shape) == 1

        if self.n < self.reservoir_size:
            take = min(self.reservoir_size - self.n, m)
            self.res = np.concatenate([self.res, v[:take]])
            v = v[take:]
            self.n += take
        for x in v:
            self.n += 1
            j = self.rng.integers(0, self.n)
            if j < self.reservoir_size:
                self.res[j] = x

    def get(self):
        return self.res.copy()

    def __add__(self, other):
        assert isinstance(other, Dist)

        assert self.prefix == other.prefix

        new = Dist(prefix=self.prefix, reservoir_size=self.reservoir_size, rng=0)

        out.n = self.n + other.n
        both = np.concatenate([self.res, other.res])
        if both.size <= out.reservoir_size:
            out.res = both
        else:
            idx = np.random.default_rng().choice(both.size, size=out.reservoir_size, replace=False)
            out.res = both[idx]
            out.n = out.reservoir_size
        return out

class BaseLogger:

    def __init__(self, stdout=True, tqdm=True, tensorboard=False, summary_writer=None, stats_window_size=100, prefix=''):
        self.stdout = stdout
        self.tqdm = tqdm
        self.tensorboard = tensorboard
        self.summary_writer = summary_writer
        if self.tensorboard:
            assert self.summary_writer is not None
        if (self.summary_writer is not None) and (not self.tensorboard):
            warnings.warn("tensorboard is set to False but summary writer is provided, this may produce unexpected behaviour")
        self.stats_window_size = stats_window_size
        self.prefix = prefix if prefix[-1] == '/' else prefix + '/'
        self.stats = {}

    def reset(self):
        self.stats = {}

    def add(self, new):
        raise NotImplementedError

    def log(self, step):
        raise NotImplementedError

class RolloutLogger(BaseLogger):

    def __init__(self, stdout=True, tqdm=True, tensorboard=False, summary_writer=None, stats_window_size=100, prefix='rollout'):
        super().__init__(stdout=stdout, tqdm=tqdm, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=stats_window_size, prefix=prefix)

    def add(self, info):
        raise NotImplementedError("TODO")

    def log(self, step):

        stats_to_log = {}
        
        for key, val in self.stats.items():
            if len(val) > 1:
                stats_to_log[key] = np.mean(val[:-1][-self.stats_window_size:])

        if self.start_time is not None:
            current_time = time.time()
            stats_to_log['fps'] = step/(current_time - self.start_time)
    
        if self.tensorboard:
            with self.summary_writer.as_default():
                for key in stats_to_log:
                    tf.summary.scalar(self.prefix + key, stats_to_log[key], step=step)

        if self.stdout:
            stats_to_print = {key: "{0:.4g}".format(val) for key, val in stats_to_log.items()}
            if self.start_time is not None:
                stats_to_print['time_elapsed'] = "{0:.4g}".format(current_time - self.start_time)
            stats_to_print['total_timesteps'] = str(step)
            max_key_len = max([len(key) for key in stats_to_print] + [len(self.prefix) - 2])
            max_val_len = max([len(val) for val in stats_to_print.values()])
            stdout = ""
            max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
            stdout += ("-"*max_len + "\n")
            stdout += ("|  "+self.prefix + " "*(2+max_key_len-len(self.prefix)+2) + "|" + " "*(2 + max_val_len + 2)+"|\n")
            for key, val in stats_to_print.items():
                stdout += ("|    "+key + " "*(max_key_len-len(key)+2) + "|  " + val  +" "*(max_val_len - len(val) + 2)+"|\n")
            stdout += ("-"*max_len + "\n")
            if self.tqdm:
                tqdm.write(stdout)
            else:
                print(stdout)

class MetricsLogger(BaseLogger):

    def __init__(self, stdout=True, tqdm=True, tensorboard=False, summary_writer=None, stats_window_size=100, prefix='metrics'):
        super().__init__(stdout=stdout, tqdm=tqdm, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=stats_window_size, prefix=prefix)
        self.dists = {}

    def reset(self):
        super().reset()
        self.dists = {}

    def add(self, new):
        for key, val in new.items():
            if isinstance(val, Metrics):
                met = val.get()
                for k, v in met.items():
                    if k in self.stats:
                        self.stats.append(v)
                    else:
                        self.stats[k] = [v]
            elif isinstance(val, Dist):
                self.dists[key] = val.get()
            elif isinstance(val, float):
                if key in self.stats:
                    self.stats[key].append(val)
                else:
                    self.stats[key] = [val]
            else:
                raise NotImplementedError("MetricsLogger.add() does only supports types: Metrics, Dist and Float")

    def log(self, step):

        stats_to_log = {}
        dists_to_log = {}

        for key, val in self.stats.items():
            if len(val) > 0:
                stats_to_log[key] = np.mean(val[-self.stats_window_size:])

        for key, val in self.dists.items():
            if len(val) > 0:
                dists_to_log[key] = val

        if self.tensorboard:
            with self.summary_writer.as_default():
                for key in stats_to_log:
                    tf.summary.scalar(self.prefix + key, stats_to_log[key], step=step)
                for key in dists_to_log:
                    tf.summary.histogram(self.prefix + key, data=dists_to_log[key], step=step)

        if self.stdout:
            stats_to_print = {key: "{0:.4g}".format(val) for key, val in stats_to_log.items()}
            max_key_len = max([len(key) for key in stats_to_print] + [len(self.prefix) - 2])
            max_val_len = max([len(val) for val in stats_to_print.values()])
            stdout = ""
            max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
            stdout += ("-"*max_len + "\n")
            stdout += ("|  "+self.prefix + " "*(2+max_key_len-len(self.prefix)+2) + "|" + " "*(2 + max_val_len + 2)+"|\n")
            for key, val in stats_to_print.items():
                stdout += ("|    "+key + " "*(max_key_len-len(key)+2) + "|  " + val  +" "*(max_val_len - len(val) + 2)+"|\n")
            stdout += ("-"*max_len + "\n")
            if self.tqdm:
                tqdm.write(stdout)
            else:
                print(stdout)