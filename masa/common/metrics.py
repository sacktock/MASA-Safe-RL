
import tensorflow as tf
import numpy as np
import warnings
import time 
from tqdm import tqdm
from collections import deque
from typing import Any, Optional, TypeVar, Union, Callable, Dict, List, Tuple

class Stats:

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

        assert self.prefix == other.prefix, "can't add two Stats objects with different prefixes"

        new = Stats(prefix=self.prefix)
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

class StatsLogger(BaseLogger):

    def __init__(self, stdout=True, tqdm=True, tensorboard=False, summary_writer=None, stats_window_size=100, prefix=''):
        super().__init__(stdout=stdout, tqdm=tqdm, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=stats_window_size, prefix=prefix)
        self.dists = {}

    def reset(self):
        super().reset()
        self.dists = {}

    def add(self, new):
        for key, val in new.items():
            if isinstance(val, Stats):
                met = val.get()
                for k, v in met.items():
                    if k in self.stats:
                        self.stats.append(v)
                    else:
                        self.stats[k] = deque([v], maxlen=self.stats_window_size)
            elif isinstance(val, Dist):
                self.dists[key] = val.get()
            elif isinstance(val, float):
                if key in self.dists:
                    self.stats[key].append(val)
                else:
                    self.stats[key] = deque([val], maxlen=self.stats_window_size)
            else:
                raise NotImplementedError("StatsLogger.add() only supports types: Stats, Dist and Float")

    def log(self, step):
        self._create_logs()
        if self.tensorboard:
            self._log_to_tensorboard(step)
        if self.stdout:
            self._log_to_stdout(step)

    def _create_logs(self):
        self._create_stats_to_log()
        self._create_dists_to_log()

    def _create_stats_to_log(self):
        self.stats_to_log = {}
        for key, val in self.stats.items():
            if len(val) > 0:
                self.stats_to_log[key] = np.mean(val)

    def _create_dists_to_log(self):
        self.dists_to_log = {}
        for key, val in self.dists.items():
            if len(val) > 0:
                self.dists_to_log[key] = val

    def _log_to_tensorboard(self, step):
        assert self.summary_writer is not None, "You're trying to log to tensorboard without a summary writer setup!"
        with self.summary_writer.as_default():
            for key in self.stats_to_log:
                tf.summary.scalar(self.prefix + key, self.stats_to_log[key], step=step)
            for key in self.dists_to_log:
                tf.summary.histogram(self.prefix + key, data=self.dists_to_log[key], step=step)

    def _log_to_stdout(self, step):
        stats_to_print = {key: "{0:.4g}".format(val) for key, val in self.stats_to_log.items()}
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

class RolloutLogger(BaseLogger):

    def __init__(self, stdout=True, tqdm=True, tensorboard=False, summary_writer=None, stats_window_size=100, prefix=''):
        super().__init__(stdout=stdout, tqdm=tqdm, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=stats_window_size, prefix=prefix)
        self.start_time = None

    def add(self, info, verbose=0):
        if self.start_time is None:
            self.start_time = time.time()
        if "episode" in info.get("constraint", {}):
            ep_metrics = info["constraint"]["episode"]
            self._add_scalars(ep_metrics)
        if "episode" in info.get("metrics", {}):
            ep_metrics = info["metrics"]["episode"]
            self._add_scalars(ep_metrics)

    def log(self, step):
        self._create_logs()
        if self.tensorboard:
            self._log_to_tensorboard(step)
        if self.stdout:
            self._log_to_stdout(step)

    def _create_logs(self):
        self._create_stats_to_log()

    def _add_scalars(self, scalars):
        for k, v in scalars.items():
            if k in self.stats.keys():
                self.stats[k].append(v)
            else:
                self.stats[k] = deque([v], maxlen=self.stats_window_size+1)
                    
    def _create_stats_to_log(self):
        self.stats_to_log = {}
        for key, val in self.stats.items():
            if len(val) > 1:
                item = val.pop()
                self.stats_to_log[key] = np.mean(val)
                val.append(item)

    def _log_to_tensorboard(self, step):
        assert self.summary_writer is not None, "You're trying to log to tensorboard without a summary writer setup!"
        with self.summary_writer.as_default():
            for key in self.stats_to_log:
                tf.summary.scalar(self.prefix + key, self.stats_to_log[key], step=step)

    def _log_to_stdout(self, step):
        stats_to_print = {key: "{0:.4g}".format(val) for key, val in self.stats_to_log.items()}
        if self.start_time is not None:
            current_time = time.time()
            stats_to_print['fps'] = "{0:.4g}".format(step/(current_time - self.start_time))
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

class TrainLogger(BaseLogger):

    def __init__(
        self, 
        loggers: List[Tuple[str, BaseLogger]],
        stdout: bool = True, 
        tqdm: bool = True,
        tensorboard: bool = False,
        summary_writer: bool = None,
        stats_window_size: Union[int, List[int]] = 100,
        prefix: str = ''
    ):
        self.loggers = {}
        self.stdout = stdout
        self.tqdm = tqdm
        self.tensorboard = tensorboard
        self.summary_writer = summary_writer
        self.prefix = prefix

        if isinstance(stats_window_size, int):
            self.stats_window_size = [stats_window_size]*len(loggers)
        elif isinstance(stats_window_size, list):
            self.stats_window_size = stats_window_size
        else:
            raise RuntimeError(
                "Expected type int or List[int] for stats_window_size"
            )

        for idx, (key, ctor) in enumerate(loggers):
            self.loggers[key] = ctor(
                stdout=self.stdout, 
                tqdm=self.tqdm, 
                tensorboard=self.tensorboard, 
                summary_writer=self.summary_writer, 
                stats_window_size=self.stats_window_size[idx],
                prefix=key,
                
            )

        self.start_time = None

    def add(self, key, obj):
        if self.start_time is None:
            self.start_time = time.time()
        self.loggers[key].add(obj)

    def log(self, step):
        for key in self.loggers.keys():
            self.loggers[key]._create_logs()
            if self.tensorboard:
                self.loggers[key]._log_to_tensorboard(step)

        if self.stdout:
            self._log_to_stdout(step)

    def _log_to_stdout(self, step):
        stats_to_print = {}
        stats_to_print['run'] = {}
        if self.start_time is not None:
            current_time = time.time()
            stats_to_print['run']['fps'] = "{0:.4g}".format(step/(current_time - self.start_time))
            stats_to_print['run']['time_elapsed'] = "{0:.4g}".format(current_time - self.start_time)
        stats_to_print['run']['total_timesteps'] = str(step)
        for key in self.loggers.keys():
            stats_to_print[key] = {k: "{0:.4g}".format(v) for k, v in self.loggers[key].stats_to_log.items()}
        max_key_len = 0
        max_val_len = 0
        for key in stats_to_print.keys():
            if not stats_to_print[key]:
                continue
            max_key_len = max(max_key_len, max([len(k) for k in stats_to_print[key].keys()] + [len(self.prefix) - 2]))
            max_val_len = max(max_val_len, max([len(v) for v in stats_to_print[key].values()]))
        stdout = ""
        max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
        stdout += ("-"*max_len + "\n")
        if self.prefix:
            stdout += ("|  "+self.prefix + " "*(2+max_key_len-len(self.prefix)+2) + "|" + " "*(2 + max_val_len + 2)+"|\n")
            stdout += ("-"*max_len + "\n")
        for key in stats_to_print.keys():
            if not stats_to_print[key]:
                continue
            prefix = key+"/"
            stdout += ("|  "+prefix+" "*(2+max_key_len-len(prefix)+2) + "|" + " "*(2 + max_val_len + 2)+"|\n")
            for k, v in stats_to_print[key].items():
                stdout += ("|    "+k + " "*(max_key_len-len(k)+2) + "|  " + v  +" "*(max_val_len - len(v) + 2)+"|\n")
            stdout += ("-"*max_len + "\n")
        if self.tqdm:
            tqdm.write(stdout)
        else:
            print(stdout)
            

            



            

