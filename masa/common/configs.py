import re
import sys
import io
import json

import contextlib
import glob
import os
import re
import shutil


class Path:

  filesystems = []

  def __new__(cls, path):
    path = str(path)
    for impl, pred in cls.filesystems:
      if pred(path):
        obj = super().__new__(impl)
        obj.__init__(path)
        return obj
    raise NotImplementedError(f'No filesystem supports: {path}')

  def __getnewargs__(self):
    return (self._path,)

  def __init__(self, path):
    assert isinstance(path, str)
    path = re.sub(r'^\./*', '', path)  # Remove leading dot or dot slashes.
    path = re.sub(r'(?<=[^/])/$', '', path)  # Remove single trailing slash.
    path = path or '.'  # Empty path is represented by a dot.
    self._path = path

  def __truediv__(self, part):
    sep = '' if self._path.endswith('/') else '/'
    return type(self)(f'{self._path}{sep}{str(part)}')

  def __repr__(self):
    return f'Path({str(self)})'

  def __fspath__(self):
    return str(self)

  def __eq__(self, other):
    return self._path == other._path

  def __lt__(self, other):
    return self._path < other._path

  def __str__(self):
    return self._path

  @property
  def parent(self):
    if '/' not in self._path:
      return type(self)('.')
    parent = self._path.rsplit('/', 1)[0]
    parent = parent or ('/' if self._path.startswith('/') else '.')
    return type(self)(parent)

  @property
  def name(self):
    if '/' not in self._path:
      return self._path
    return self._path.rsplit('/', 1)[1]

  @property
  def stem(self):
    return self.name.split('.', 1)[0] if '.' in self.name else self.name

  @property
  def suffix(self):
    return ('.' + self.name.split('.', 1)[1]) if '.' in self.name else ''

  def read(self, mode='r'):
    assert mode in 'r rb'.split(), mode
    with self.open(mode) as f:
      return f.read()

  def write(self, content, mode='w'):
    assert mode in 'w a wb ab'.split(), mode
    with self.open(mode) as f:
      f.write(content)

  @contextlib.contextmanager
  def open(self, mode='r'):
    raise NotImplementedError

  def absolute(self):
    raise NotImplementedError

  def glob(self, pattern):
    raise NotImplementedError

  def exists(self):
    raise NotImplementedError

  def isfile(self):
    raise NotImplementedError

  def isdir(self):
    raise NotImplementedError

  def mkdirs(self):
    raise NotImplementedError

  def remove(self):
    raise NotImplementedError

  def rmtree(self):
    raise NotImplementedError

  def copy(self, dest):
    raise NotImplementedError

  def move(self, dest):
    self.copy(dest)
    self.remove()


class LocalPath(Path):

  def __init__(self, path):
    super().__init__(os.path.expanduser(str(path)))

  @contextlib.contextmanager
  def open(self, mode='r'):
    with open(str(self), mode=mode) as f:
      yield f

  def absolute(self):
    return type(self)(os.path.absolute(str(self)))

  def glob(self, pattern):
    for path in glob.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return os.path.exists(str(self))

  def isfile(self):
    return os.path.isfile(str(self))

  def isdir(self):
    return os.path.isdir(str(self))

  def mkdirs(self):
    os.makedirs(str(self), exist_ok=True)

  def remove(self):
    os.rmdir(str(self)) if self.isdir() else os.remove(str(self))

  def rmtree(self):
    shutil.rmtree(self)

  def copy(self, dest):
    if self.isfile():
      shutil.copy(self, type(self)(dest))
    else:
      shutil.copytree(self, type(self)(dest), dirs_exist_ok=True)

  def move(self, dest):
    shutil.move(self, dest)


class GFilePath(Path):

  def __init__(self, path):
    path = str(path)
    if not (path.startswith('/') or '://' in path):
      path = os.path.abspath(os.path.expanduser(path))
    super().__init__(path)
    import tensorflow as tf
    self._gfile = tf.io.gfile

  @contextlib.contextmanager
  def open(self, mode='r'):
    path = str(self)
    if 'a' in mode and path.startswith('/cns/'):
      path += '%r=3.2'
    if mode.startswith('x') and self.exists():
      raise FileExistsError(path)
      mode = mode.replace('x', 'w')
    with self._gfile.GFile(path, mode) as f:
      yield f

  def absolute(self):
    return self

  def glob(self, pattern):
    for path in self._gfile.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return self._gfile.exists(str(self))

  def isfile(self):
    return self.exists() and not self.isdir()

  def isdir(self):
    return self._gfile.isdir(str(self))

  def mkdirs(self):
    self._gfile.makedirs(str(self))

  def remove(self):
    self._gfile.remove(str(self))

  def rmtree(self):
    self._gfile.rmtree(str(self))

  def copy(self, dest):
    self._gfile.copy(str(self), str(dest), overwrite=True)

  def move(self, dest):
    dest = Path(dest)
    if dest.isdir():
      dest.rmtree()
    self._gfile.rename(self, str(dest), overwrite=True)


Path.filesystems = [
    (GFilePath, lambda path: path.startswith('gs://')),
    (GFilePath, lambda path: path.startswith('/cns/')),
    (LocalPath, lambda path: True),
]

class Flags:

  def __init__(self, *args, **kwargs):
    self._config = Config(*args, **kwargs)

  def parse(self, argv=None, help_exists=True):
    parsed, remaining = self.parse_known(argv)
    for flag in remaining:
      if flag.startswith('--'):
        raise ValueError(f"Flag '{flag}' did not match any config keys.")
    #assert not remaining or , remaining
    return parsed

  def parse_known(self, argv=None, help_exists=False):
    if argv is None:
      argv = sys.argv[1:]
    if '--help' in argv:
      print('\nHelp:')
      lines = str(self._config).split('\n')[2:]
      print('\n'.join('--' + re.sub(r'[:,\[\]]', '', x) for x in lines))
      help_exists and sys.exit()
    parsed = {}
    remaining = []
    key = None
    vals = None
    for arg in argv:
      if arg.startswith('--'):
        if key:
          self._submit_entry(key, vals, parsed, remaining)
        if '=' in arg:
          key, val = arg.split('=', 1)
          vals = [val]
        else:
          key, vals = arg, []
      else:
        if key:
          vals.append(arg)
        else:
          remaining.append(arg)
    self._submit_entry(key, vals, parsed, remaining)
    parsed = self._config.update(parsed)
    return parsed, remaining

  def _submit_entry(self, key, vals, parsed, remaining):
    if not key and not vals:
      return
    if not key:
      vals = ', '.join(f"'{x}'" for x in vals)
      raise ValueError(f"Values {vals} were not preceded by any flag.")
    name = key[len('--'):]
    if '=' in name:
      remaining.extend([key] + vals)
      return
    if self._config.IS_PATTERN.fullmatch(name):
      pattern = re.compile(name)
      keys = {k for k in self._config.flat if pattern.fullmatch(k)}
    elif name in self._config:
      keys = [name]
    else:
      keys = []
    if not keys:
      remaining.extend([key] + vals)
      return
    if not vals:
      raise ValueError(f"Flag '{key}' was not followed by any values.")
    for key in keys:
      parsed[key] = self._parse_flag_value(self._config[key], vals, key)

  def _parse_flag_value(self, default, value, key):
    value = value if isinstance(value, (tuple, list)) else (value,)
    if isinstance(default, (tuple, list)):
      if len(value) == 1 and ',' in value[0]:
        value = value[0].split(',')
      return tuple(self._parse_flag_value(default[0], [x], key) for x in value)
    assert len(value) == 1, value
    value = str(value[0])
    if default is None:
      return value
    if isinstance(default, bool):
      try:
        return bool(['False', 'True'].index(value))
      except ValueError:
        message = f"Expected bool but got '{value}' for key '{key}'."
        raise TypeError(message)
    if isinstance(default, int):
      try:
        value = float(value)  # Allow scientific notation for integers.
        assert float(int(value)) == value
      except (TypeError, AssertionError):
        message = f"Expected int but got float '{value}' for key '{key}'."
        raise TypeError(message)
      return int(value)
    if isinstance(default, dict):
      raise TypeError(
          f"Key '{key}' refers to a whole dict. Please speicfy a subkey.")
    return type(default)(value)


class Config(dict):

  SEP = '.'
  IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

  def __init__(self, *args, **kwargs):
    mapping = dict(*args, **kwargs)
    mapping = self._flatten(mapping)
    mapping = self._ensure_keys(mapping)
    mapping = self._ensure_values(mapping)
    self._flat = mapping
    self._nested = self._nest(mapping)
    # Need to assign the values to the base class dictionary so that
    # conversion to dict does not lose the content.
    super().__init__(self._nested)

  @property
  def flat(self):
    return self._flat.copy()

  def save(self, filename):
    filename = path.Path(filename)
    if filename.suffix == '.json':
      filename.write(json.dumps(dict(self)))
    elif filename.suffix in ('.yml', '.yaml'):
      import ruamel.yaml as yaml
      with io.StringIO() as stream:
        yaml.safe_dump(dict(self), stream)
        filename.write(stream.getvalue())
    else:
      raise NotImplementedError(filename.suffix)

  @classmethod
  def load(cls, filename):
    filename = path.Path(filename)
    if filename.suffix == '.json':
      return cls(json.loads(filename.read_text()))
    elif filename.suffix in ('.yml', '.yaml'):
      import ruamel.yaml as yaml
      return cls(yaml.safe_load(filename.read_text()))
    else:
      raise NotImplementedError(filename.suffix)

  def __contains__(self, name):
    try:
      self[name]
      return True
    except KeyError:
      return False

  def __getattr__(self, name):
    if name.startswith('_'):
      return super().__getattr__(name)
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __getitem__(self, name):
    result = self._nested
    for part in name.split(self.SEP):
      try:
        result = result[part]
      except TypeError:
        raise KeyError
    if isinstance(result, dict):
      result = type(self)(result)
    return result

  def __setattr__(self, key, value):
    if key.startswith('_'):
      return super().__setattr__(key, value)
    message = f"Tried to set key '{key}' on immutable config. Use update()."
    raise AttributeError(message)

  def __setitem__(self, key, value):
    if key.startswith('_'):
      return super().__setitem__(key, value)
    message = f"Tried to set key '{key}' on immutable config. Use update()."
    raise AttributeError(message)

  def __reduce__(self):
    return (type(self), (dict(self),))

  def __str__(self):
    lines = ['\nConfig:']
    keys, vals, typs = [], [], []
    for key, val in self.flat.items():
      keys.append(key + ':')
      vals.append(self._format_value(val))
      typs.append(self._format_type(val))
    max_key = max(len(k) for k in keys) if keys else 0
    max_val = max(len(v) for v in vals) if vals else 0
    for key, val, typ in zip(keys, vals, typs):
      key = key.ljust(max_key)
      val = val.ljust(max_val)
      lines.append(f'{key}  {val}  ({typ})')
    return '\n'.join(lines)

  def update(self, *args, **kwargs):
    result = self._flat.copy()
    inputs = self._flatten(dict(*args, **kwargs))
    for key, new in inputs.items():
      if self.IS_PATTERN.match(key):
        pattern = re.compile(key)
        keys = {k for k in result if pattern.match(k)}
      else:
        keys = [key]
      if not keys:
        raise KeyError(f'Unknown key or pattern {key}.')
      for key in keys:
        old = result[key]
        try:
          if isinstance(old, int) and isinstance(new, float):
            if float(int(new)) != new:
              message = f"Cannot convert fractional float {new} to int."
              raise ValueError(message)
          result[key] = type(old)(new)
        except (ValueError, TypeError):
          raise TypeError(
              f"Cannot convert '{new}' to type '{type(old).__name__}' " +
              f"for key '{key}' with previous value '{old}'.")
    return type(self)(result)

  def _flatten(self, mapping):
    result = {}
    for key, value in mapping.items():
      if isinstance(value, dict):
        for k, v in self._flatten(value).items():
          if self.IS_PATTERN.match(key) or self.IS_PATTERN.match(k):
            combined = f'{key}\\{self.SEP}{k}'
          else:
            combined = f'{key}{self.SEP}{k}'
          result[combined] = v
      else:
        result[key] = value
    return result

  def _nest(self, mapping):
    result = {}
    for key, value in mapping.items():
      parts = key.split(self.SEP)
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _ensure_keys(self, mapping):
    for key in mapping:
      assert not self.IS_PATTERN.match(key), key
    return mapping

  def _ensure_values(self, mapping):
    result = json.loads(json.dumps(mapping))
    for key, value in result.items():
      if isinstance(value, list):
        value = list(value)
      if isinstance(value, tuple):
        if len(value) == 0:
          message = 'Empty lists are disallowed because their type is unclear.'
          raise TypeError(message)
        if not isinstance(value[0], (str, float, int, bool)):
          message = 'Lists can only contain strings, floats, ints, bools'
          message += f' but not {type(value[0])}'
          raise TypeError(message)
        if not all(isinstance(x, type(value[0])) for x in value[1:]):
          message = 'Elements of a list must all be of the same type.'
          raise TypeError(message)
      result[key] = value
    return result

  def _format_value(self, value):
    if isinstance(value, (list, tuple)):
      return '[' + ', '.join(self._format_value(x) for x in value) + ']'
    return str(value)

  def _format_type(self, value):
    if isinstance(value, (list, tuple)):
      assert len(value) > 0, value
      return self._format_type(value[0]) + 's'
    return str(type(value).__name__)
