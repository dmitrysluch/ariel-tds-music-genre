import inspect
import numpy as np
import numba as nb
import pandas as pd
import os
import uuid
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import deque
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Type, TypeVar
)

# Looks like my laptop is using swap file as crazy. 
# Select the value to be about page size, really do not want 
# to go over that value.
MAX_BATCH_SZ_BYTES = 4096


def is_jupyter() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False


if is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ------------------------------------------------------------------
#  LazyFeature: holds data in memory or a path on disk
# ------------------------------------------------------------------
class MemoryManager:
    def __init__(self, max_mem=10 * 1024 * 1024 * 1024):
        self._q = deque()
        self._max_mem=max_mem
        self._used_mem = 0
    def up(self, element: Tuple[LazyFeature, int, int]):
        # Ups element in LRU cache
        if element in self._q:
            self._q.remove(element)  # Remove from current position
        else:
            self._used_mem += element[2]
        self._q.append(element)      # Add to end (most recently used)
    def cleanup(self):
        # Dumps element to disk. Element is tuple (lazy_feature, batch_index, batch_size in bytes) 
        while self._used_mem > self._max_mem and len(self._q) > 0:
            # Get the least recently used element
            lf, batch_idx, batch_sz = self._q.popleft()
            # Dump it to disk
            lf.dump_batch(batch_idx)
            # Update memory usage
            self._used_mem -= batch_sz

class LazyFeatureIterator:
    def __init__(self, lazy_feature: LazyFeature):
        self._lazy_feature = lazy_feature
        self._current_batch = -1

    def __next__(self):
        self._current_batch += 1
        if self._current_batch >= len(self._lazy_feature._data):
            raise StopIteration
            
        batch_idx = self._current_batch
        
        if self._lazy_feature._data[batch_idx] is None:
            self._lazy_feature.load_batch(batch_idx)
        
        batch = self._lazy_feature._data[batch_idx]
        batch_sz_bytes = batch.nbytes if isinstance(batch, np.array) else self._lazy_feature._batch_sz_bytes
        self._lazy_feature._memory_manager.up((self._lazy_feature, batch_idx, batch_sz_bytes))
        self._lazy_feature._memory_manager.cleanup()
        
        return batch

class LazyFeature:
    """
    Wraps a feature's data, allowing it to be dumped to disk and reloaded on demand.
    """

    def __init__(self, memory_manager: MemoryManager, data: list[Any], batch_size: int, path: str, batch_sz_bytes=1024):
        """
        If batch_size > 1, data must be a list of np.ndarrays of shape
        [b, ...], where b = batch_size for all items except maybe the last which may be smaller.
        Path must be set and will be used for dumping when nessesary.
        """
        self._memory_manager = memory_manager
        self._data = data
        self._batch_size = batch_size
        self._path = path
        self._batch_sz_bytes = batch_sz_bytes
        for i in len(self._data):
            batch_sz_bytes_calc = self._data[i].nbytes if isinstance(self._data[i], np.ndarray) else batch_sz_bytes
            self._memory_manager.up((self, i, batch_sz_bytes_calc))

    @staticmethod
    def from_numpy(memory_manager: MemoryManager, arr: np.ndarray, path: str) -> LazyFeature:
        # Determine the size of the batch, using array.nbytes for first element
        # and MAX_BATCH_SZ_BYTES, split data accordingly and initialize in-memory lazy feature
        batch_nbytes = arr[0].nbytes if len(arr) > 0 else MAX_BATCH_SZ_BYTES
        batch_size = max(1, MAX_BATCH_SZ_BYTES // batch_nbytes)
        
        # Split array into batches
        batches = np.array_split(arr, (arr.shape[0] + batch_size - 1) // batch_size, axis=0)
        
        # Create LazyFeature with batches
        return LazyFeature(
            memory_manager=memory_manager,
            data=batches,
            batch_size=batch_size,
            path=path,
            batch_sz_bytes=batch_nbytes * batch_size
        )

    @property
    def batched(self) -> bool:
        return self._batch_size > 1

    def dump_batch(self, i: int):
        if self._data[i] is None:
            return
        print("Dumping:", self.path, "batch:", i)
        batch_data = self._data[i]
        batch_path = os.path.join(self._path, f"batch_{i}")
        if isinstance(batch_data, np.ndarray):
            np.save(batch_path + ".npy", batch_data)
        else:
            with open(batch_path + ".pkl", "wb") as f:
                pickle.dump(batch_data, f)
        self._data[i] = None

    def load_batch(self, i: int):
        print("Loading:", self.path, "batch:", i)
        batch_path = os.path.join(self._path, f"batch_{i}")
        if os.path.exists(batch_path + ".npy"):
            self._data[i] = np.load(batch_path + ".npy")
        elif os.path.exists(batch_path + ".pkl"):
            with open(batch_path + ".pkl", "rb") as f:
                self._data[i] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Batch file for index {i} not found.")

    def dump_all(self) -> None:
        for i in range(len(self._data)):
            self.dump_batch(i)
    
    def __iter__(self) -> LazyFeatureIterator:
        return LazyFeatureIterator(self)
    
    def __len__(self) -> int:
        return len(self._data)

# ------------------------------------------------------------------
#  Extractor-related
# ------------------------------------------------------------------

class ExtractorType(Enum):
    MAPPER = auto()
    FLAT_MAPPER = auto()
    BATCH_MAPPER = auto()
    BATCH_FLAT_MAPPER = auto()


class Extractor:
    """
    Stores parameters/metadata for a single (possibly multi-output) extractor.
    """

    def __init__(
        self,
        feature_names: List[str],
        method: Callable[..., Any],
        deps: List[str],
        extractor_type: ExtractorType,
        shuffle: bool,
        uniform: bool,
        multi_output: bool,
        seed: int,
    ):
        self.feature_names = feature_names
        self.method = method
        self.deps = deps
        self.type = extractor_type
        self.shuffle = shuffle
        self.uniform = uniform
        self.multi_output = multi_output
        self.seed = seed

    def __repr__(self) -> str:
        return (f"Extractor("
                f"features={self.feature_names}, deps={self.deps}, "
                f"type={self.type.name}, shuffle={self.shuffle}, "
                f"uniform={self.uniform}, multi_output={self.multi_output}, "
                f"seed={self.seed})")

# ------------------------------------------------------------------
#  Extractor decorators
# ------------------------------------------------------------------


def mapper(
        feature_name: Union[str, List[str]],
        *,
        shuffle: bool = False,
        uniform: bool = True,
        batched: bool = False,
        seed: int = 42):
    def decorator(method: Callable[..., Any]):
        deps = DAGXtractor._get_dependencies_from_signature(method)
        multi_output = isinstance(feature_name, list)
        f_names = feature_name if multi_output else [feature_name]

        ext_type = ExtractorType.BATCH_MAPPER if batched else ExtractorType.MAPPER
        return Extractor(
            feature_names=f_names,
            method=method,
            deps=deps,
            extractor_type=ext_type,
            shuffle=shuffle,
            uniform=uniform,
            multi_output=multi_output,
            seed=seed,  # or keep seed param if you want
        )
    return decorator


def flat_mapper(
    feature_name: Union[str, List[str]],
    *,
    shuffle: bool = False,
    uniform: bool = True,
    batched: bool = False,
    seed: int = 42
):
    def decorator(method: Callable[..., Any]):
        deps = DAGXtractor._get_dependencies_from_signature(method)
        multi_output = isinstance(feature_name, list)
        f_names = feature_name if multi_output else [feature_name]

        ext_type = ExtractorType.BATCH_FLAT_MAPPER if batched else ExtractorType.FLAT_MAPPER
        return Extractor(
            feature_names=f_names,
            method=method,
            deps=deps,
            extractor_type=ext_type,
            shuffle=shuffle,
            uniform=uniform,
            multi_output=multi_output,
            seed=seed,
        )
    return decorator

# ------------------------------------------------------------------
#  Table class
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    id: uuid.UUID = field(init=False)
    parent_index_mapping: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'id', uuid.uuid4())

# ------------------------------------------------------------------
#  Metaclass for DAGXtractor
# ------------------------------------------------------------------


class DAGXtractorMeta(type):
    """
    Metaclass that sets up class-level extractor dictionaries.
    The pipeline is shared by all instances of the class.
    """
    def __new__(mcs, name: str, bases: Tuple[type, ...], dct: dict) -> "DAGXtractorMeta":
        extractors = [v for k, v in dct.items() if isinstance(v, Extractor)]
        dct = {k: v for k, v in dct.items() if not isinstance(v, Extractor)}
        cls = super().__new__(mcs, name, bases, dct)
        # Initialize or reset class-level dicts
        cls._extractors = {}
        cls._extractors_for_feature = {}
        for ext in extractors:
            cls._store_extractor(ext)
        return cls

# ------------------------------------------------------------------
#  DAGXtractor
# ------------------------------------------------------------------


class DAGXtractor(metaclass=DAGXtractorMeta):
    """
    A DAG-based pipeline with class-level extractor definitions (registered via classmethods),
    and per-instance data storage using LazyFeature wrappers.
    """

    def __init__(self, seed: int = 42, batch_size=64, path=None, **initial_features: Any):
        """
        :param seed: Seed for controlling the random number generation (used for shuffling).
        :param initial_features: key=feature_name, value=either a single np.ndarray, a list, or scalar, etc.
        """
        self._seed = seed
        self._batch_size = batch_size
        # features & indexes are instance-level
        self.features: Dict[str, LazyFeature] = {}
        self.tables: Dict[str, Table] = {}
        if path is None:
            path = os.path.join(".dagxtractor", uuid.uuid4().hex)
        self._path = path
        os.makedirs(self._path, exist_ok=True)

        # Initialize from user-provided features
        for feat_name, feat_value in initial_features.items():
            self._initialize_feature(feat_name, feat_value)

    # region: Register Extractors (as classmethods)

    @classmethod
    def register_mapper(
        cls,
        feature_name: Union[str, List[str]],
        method: Callable[..., Any],
        *,
        shuffle: bool = False,
        uniform: bool = True,
        batched: bool = False,
        seed: int = 42
    ) -> None:
        """
        Register a mapper that produces one or more features from dependencies.
        This is stored at the class level, so all future instances share the pipeline definition.
        Multi-output mappers must produce dict of values corresponding to output features.
        Multi-output batched mappers must produce dict of np.arrays corresponding to output features.
        """
        deps = cls._get_dependencies_from_signature(method)
        multi_output = isinstance(feature_name, list)
        f_names = feature_name if multi_output else [feature_name]

        if batched and not uniform:
            raise ValueError("Batching only supported for uniform mappers.")

        ext_type = ExtractorType.BATCH_MAPPER if batched else ExtractorType.MAPPER
        ext = Extractor(
            feature_names=f_names,
            method=method,
            deps=deps,
            extractor_type=ext_type,
            shuffle=shuffle,
            uniform=uniform,
            multi_output=multi_output,
            seed=seed,  # or keep seed param if you want
        )
        # Register
        cls._store_extractor(ext)

    @classmethod
    def register_flat_mapper(
        cls,
        feature_name: Union[str, List[str]],
        method: Callable[..., Any],
        *,
        shuffle: bool = False,
        uniform: bool = True,
        batched: bool = False,
        seed: int = 42
    ) -> None:
        """
        Similar to register_mapper but for flat mapping (explode samples).
        """
        deps = cls._get_dependencies_from_signature(method)
        multi_output = isinstance(feature_name, list)
        f_names = feature_name if multi_output else [feature_name]

        if batched and not uniform:
            raise ValueError(
                "Batching only supported for uniform flatmappers.")

        ext_type = ExtractorType.BATCH_FLAT_MAPPER if batched else ExtractorType.FLAT_MAPPER
        ext = Extractor(
            feature_names=f_names,
            method=method,
            deps=deps,
            extractor_type=ext_type,
            shuffle=shuffle,
            uniform=uniform,
            multi_output=multi_output,
            seed=seed,
        )
        cls._store_extractor(ext)

    @classmethod
    def _store_extractor(cls, ext: Extractor) -> None:
        """
        Helper: store an Extractor in class-level dictionaries.
        The "primary" feature is ext.feature_names[0].
        All features in ext.feature_names map to the same Extractor instance.
        """
        primary = ext.feature_names[0]
        if primary in cls._extractors:
            raise ValueError(
                f"Primary feature '{primary}' already has an extractor.")

        cls._extractors[primary] = ext
        for fn in ext.feature_names:
            if fn in cls._extractors_for_feature:
                raise ValueError(
                    f"Feature '{fn}' is already produced by another extractor.")
            cls._extractors_for_feature[fn] = ext

    @staticmethod
    def _get_dependencies_from_signature(
        method: Callable[..., Any],
        ignore: Tuple[str, ...] = ("self",)
    ) -> List[str]:
        """
        Inspect the method signature to collect parameter names (i.e. which features it needs).
        By default we ignore 'self' for bound methods, but can ignore more.
        """
        sig = inspect.signature(method)
        deps = []
        for name, _param in sig.parameters.items():
            if name not in ignore:
                deps.append(name)
        return deps

    # endregion

    # region: Pipeline methods

    def full_extract(self) -> None:
        """
        Recompute *all* features from scratch, in topological order.
        If an extractor is multi-output, a single call populates all of its features.
        """
        for feat in list(self.features.keys()):
            if feat in type(self)._extractors_for_feature:
                del self.features[feat]
                del self.tables[feat]
        self.soft_extract()

    def soft_extract(self) -> None:
        """
        Compute only missing features (i.e., those not present or empty), in topological order.
        """
        graph = self._build_dependency_graph()
        sorted_feats = self._topological_sort(graph)
        for feat in sorted_feats:
            self._compute_feature(feat)

    def dump_feature(self, feature_name: str) -> None:
        """
        Dump a specific feature to disk, clearing memory usage for that feature.
        The next time we need it, it will be loaded automatically.
        """
        if feature_name not in self.features:
            raise ValueError(
                f"Feature '{feature_name}' does not exist or not computed.")
        path = os.path.join(self._path, f"{feature_name}")
        self.features[feature_name].dump(path)

    # endregion

    # region: Internal extraction logic

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Adjacency list: each feature -> list of features it depends on (from the class-level extractors).
        """
        graph: Dict[str, List[str]] = {}
        all_feats = set(self.features.keys())  # instance-level initial
        # Also add all extractor outputs
        for ext in type(self)._extractors.values():
            all_feats.update(ext.feature_names)
            all_feats.update(ext.deps)

        # Build adjacency
        for feat in all_feats:
            if feat in type(self)._extractors_for_feature:
                ext = type(self)._extractors_for_feature[feat]
                graph[feat] = ext.deps
            else:
                # no extractor => no dependencies
                graph[feat] = []
        return graph

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Return a topological ordering of the features.
        """
        order: List[str] = []
        # 0=unvisited,1=visiting,2=visited
        colors: Dict[str, int] = defaultdict(int)
        for feat in graph:
            if colors[feat] == 0:
                self._dfs(graph, colors, order, feat)
        return order

    def _dfs(self, graph: Dict[str, List[str]], colors: Dict[str, int], order: List[str], node: str) -> None:
        if colors[node] == 1:
            raise ValueError(
                f"Cycle detected in dependency graph near '{node}'")
        if colors[node] == 2:
            return
        colors[node] = 1
        for dep in graph[node]:
            self._dfs(graph, colors, order, dep)
        colors[node] = 2
        order.append(node)

    def _compute_feature(self, feat_name: str) -> None:
        """
        Actually compute the feature if it's missing, by calling the associated extractor.
        If multi-output, that single call fills all outputs for the extractor.
        """
        # Already computed? skip. It might be an extractor returning several features
        if feat_name in self.features:
            return
        # If no extractor, we do nothing (it's presumably an initial feature or truly absent)
        if feat_name not in type(self)._extractors_for_feature:
            return

        ext = type(self)._extractors_for_feature[feat_name]

        # Check table references among dependencies
        dep_tables = [self.tables[d] for d in ext.deps]
        base_table = dep_tables[0]
        for idx in dep_tables[1:]:
            if idx != base_table:
                raise ValueError(
                    f"Dependencies of '{feat_name}' must share the same table."
                    f"Use join, to join features from different tables"
                )

        outputs, table = self._run_extractor(ext)
        # There will be features and possibly newly created parent mappings.
        for k, v in outputs.items():
            self.features[k] = LazyFeature(data=v)
            self.tables[k] = table

    @staticmethod
    def create_vectorized_mapper(ext: Extractor, dependencies: list[Any]):
        """
        Creates a function that maps `ext.method` along axis=0 of the given dependencies.
        """

        def is_numba_compatible(dep: np.ndarray | list):
            """
            You can adapt this checker as you see fit:
            - For instance, if dep is a list of objects or an ndarray with dtype=object,
              we consider it "non-numba-compatible".
            """
            return isinstance(dep, np.ndarray) and dep.dtype != object

        must_plain_loop = (not ext.uniform) or any(
            not is_numba_compatible(dep) for dep in dependencies)

        def plain_python_wrapper(*args):
            """
            Applies 'method' sample-by-sample in pure Python, returning either
            a list of results or a dict of lists of results (if multi_output).
            """
            n = len(args[0])

            if ext.multi_output:
                first_out = ext.method(*[arg[0] for arg in args])
                if not isinstance(first_out, dict):
                    raise ValueError(
                        "method is expected to return a dict when multi_output=True")
                out_dict = {
                    k: np.empty((n,) if np.isscalar(first_out) else (
                        n,) + v.shape, dtype=np.result_type(v), order='C')
                    if ext.uniform else [None] * n for k, v in first_out.items()
                }
                for i in range(1, n):
                    result = ext.method(*[arg[i] for arg in args])
                    for k in result.keys():
                        out_dict[k][i] = result[k]
                return out_dict
            else:
                # Single output
                first_out = ext.method(*[arg[0] for arg in args])
                results = np.empty((n,) if np.isscalar(first_out) else (
                    n,) + first_out.shape, dtype=np.result_type(first_out), order='C') if ext.uniform else [None] * n
                results[0] = first_out
                for i in tqdm(range(1, n), desc=ext.feature_names[0], total=n-1):
                    results[i] = ext.method(*[arg[i] for arg in args])
                return results

        if must_plain_loop:
            return plain_python_wrapper
        return plain_python_wrapper  # TODO: Use guvectorize here

    @staticmethod
    def create_vectorized_flat_mapper(ext: Extractor, dependencies: list[Any]):
        """
        Creates a function that flap maps `ext.method` along axis=0 of the given dependencies.
        """

        def plain_python_wrapper():
            """
            Applies 'method' sample-by-sample in pure Python, returning either
            a list of results or a dict of lists of results (if multi_output).
            """
            n = len(dependencies[0])
            index = []
            curr_index = 0
            if ext.multi_output:
                first_out = ext.method(*_build_sample_args(0))
                if not isinstance(first_out, dict):
                    raise ValueError(
                        "method is expected to return a dict when multi_output=True")
                out_dict = {k: [] for k in first_out.keys()}
                for i in range(1, n):
                    result = ext.method(*_build_sample_args(i))
                    for k in result.keys():
                        out_dict[k].append(result[k])
                curr_index += out_dict[iter(next(out_dict.keys()))].shape[0]
                index.append(curr_index)
                for k in result.keys():
                    if ext.uniform:
                        out_dict[k] = np.stack(out_dict[k], axis=0)
                    else:
                        out_dict[k] = sum(out_dict[k])
                return out_dict
            else:
                # Single output
                first_out = ext.method(*_build_sample_args(0))
                results = [first_out]
                for i in range(1, n):
                    results.append(ext.method(*_build_sample_args(i)))
                    curr_index += results[-1].shape[0]
                    index.append(curr_index)
                if ext.uniform:
                    results = np.stack(results, axis=0)
                else:
                    results = sum(results)
                return results

        return plain_python_wrapper

    def _run_extractor(self, ext: Extractor) -> tuple[dict[str, Any], Table]:
        # Load dependency data
        dep_data = []
        for d in ext.deps:
            if d not in self.features:
                raise ValueError(
                    f"Dependency '{d}' not found in self.features.")
            dep_data.append(self.features[d].get())

        if ext.type == ExtractorType.BATCH_MAPPER or ext.type == ExtractorType.BATCH_FLAT_MAPPER:
            batched_method = ext.method
        elif ext.type == ExtractorType.MAPPER:
            batched_method = self.create_vectorized_mapper(
                ext, dep_data)
        elif ext.type == ExtractorType.FLAT_MAPPER:
            batched_method = self.create_vectorized_flat_mapper(
                ext, dep_data)
        else:
            raise NotImplementedError()

        start = datetime.now()
        print(f"Running extractor for features",
              ', '.join(ext.feature_names), "at", start)
        try:
            result = batched_method(*dep_data)
        except Exception as e:
            raise RuntimeError(
                f"Extractor '{ext.feature_names}' failed."
            ) from e
        end = datetime.now()
        print(f"Done running extractor", "at", end,
              "total running time", end - start)

        if ext.type == ExtractorType.FLAT_MAPPER or ext.type == ExtractorType.BATCH_FLAT_MAPPER:
            result, index_borders = result
            index = np.zeros((index_borders[-1],))
            index[index_borders[:-1]] = 1
            index = np.cumsum(index)

        if not ext.multi_output:
            result = {
                ext.feature_names[0]: result
            }
        for k in result.keys():
            result[k] = np.ascontiguousarray(result[k])

        index_column_name = f"__index_{ext.feature_names[0]}"
        if ext.type == ExtractorType.FLAT_MAPPER or ext.type == ExtractorType.BATCH_FLAT_MAPPER:
            result[index_column_name] = index

        if ext.shuffle:
            if index_column_name not in result:
                result[index_column_name] = np.arange(
                    len(result[next(iter(result.keys()))]))
            rng = np.random.default_rng(
                seed=self._seed ^ hash(ext.feature_names[0]))
            self._shuffle_result(result, rng)
        table = self.tables[ext.deps[0]] if index_column_name not in result else Table(
            index_column_name)
        return result, table

    @staticmethod
    def _shuffle_result(result: dict[str, Union[list, np.ndarray]], rng: np.random.Generator) -> dict | list | np.ndarray:
        first_key = next(iter(result))
        first_val = result[first_key]
        length = len(first_val) if isinstance(
            first_val, list) else first_val.shape[0]
        perm = rng.permutation(length)

        out_dict: dict[str, Union[list, np.ndarray]] = {}
        for k, v in result.items():
            if isinstance(v, list):
                if len(v) != length:
                    raise ValueError(
                        f"Dict outputs differ in length for key '{k}'")
                out_dict[k] = [v[i] for i in perm]
            elif isinstance(v, np.ndarray):
                if v.shape[0] != length:
                    raise ValueError(
                        f"Dict outputs differ in length for key '{k}'")
                out_dict[k] = v[perm]
            else:
                raise ValueError(
                    "Unsupported type for shuffle.")
        return out_dict

    def _initialize_feature(self, feat_name: str, feat_value: Any) -> None:
        """
        Store the initial features in the pipeline as a LazyFeature and create corresponding tables.
        """
        lf = LazyFeature(feat_value)
        self.features[feat_name] = lf
        self.tables[feat_name] = Table()
    # endregion

    # region: Pandas / Numpy interface

    def numpy(self, feature_names: List[str]) -> np.ndarray:
        """
        Return a single 2D NumPy array with horizontally concatenated columns for each requested feature.
        Check that all requested features share the same index object.
        Loads any needed data from disk.
        """
        if not feature_names:
            return np.array([])

        # Check index references
        first_tlb = self.tables[feature_names[0]]
        for fn in feature_names[1:]:
            if self.tables[fn] != first_tlb:
                raise ValueError(
                    "All requested features must share the same table object for `.numpy()`.")

        arrays = []
        for fn in feature_names:
            data = self.features[fn].get()  # ensure loaded
            arr = data if isinstance(
                data, np.ndarray) else np.array(data, dtype=object)
            if arr.ndim == 1:
                # shape (N,) => interpret as single column
                arr = arr.reshape(-1, 1)
            else:
                # flatten the trailing dimensions
                arr = arr.reshape(arr.shape[0], -1)
            arrays.append(arr)
        return np.hstack(arrays)

    def get_columns(self, feature_names: List[str]) -> List[str]:
        """
        Return list of column names for the horizontally stacked data in `.numpy(feature_names)`.
        If a feature has shape (N, D), produce columns like: feat_0, feat_1, ...
        If shape is (N,), produce a single column "feat".
        """
        cols: List[str] = []
        for fn in feature_names:
            data = self.features[fn].get()
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # single column
                    length = 1
                else:
                    # flatten all but first dimension
                    length = 1
                    for d in data.shape[1:]:
                        length *= d
            else:
                # it's a list => interpret as (N,) of scalars => single column
                length = 1

            if length == 1:
                cols.append(fn)
            else:
                for i in range(length):
                    cols.append(f"{fn}_{i}")
        return cols

    def pandas(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Return a pandas DataFrame with columns from `.get_columns(...)` 
        and data from `.numpy(...)`.
        """
        arr = self.numpy(feature_names)
        col_names = self.get_columns(feature_names)
        return pd.DataFrame(arr, columns=col_names)

    def __getitem__(self, item: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(item, str):
            item = [item]
        return self.pandas(item)

    # endregion


__all__ = ["DAGXtractor", "mapper", "flat_mapper"]
