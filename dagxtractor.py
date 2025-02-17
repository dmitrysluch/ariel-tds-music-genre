import inspect
import numpy as np
import pandas as pd
import os
import uuid
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import deque
import vectorize
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Type, TypeVar, Self
)


MAX_BATCH_SZ_BYTES = 4 * 1024 * 1024 # 4Mb
DEFAULT_SCALAR_SIZE = 8
PARENT_INDEX_MAPPING_COLUMN = "__parent_index_mapping__{}"


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

# TODO: maybe use something more efficient then LRU
class MemoryManager:
    def __init__(self, max_mem=10 * 1024 * 1024 * 1024):
        self._q = deque()
        self._max_mem = max_mem
        self._used_mem = 0

    def up(self, element: Tuple["LazyFeature", int, int]):
        # Ups element in LRU cache
        if element in self._q:
            # TODO: optimize it
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
    def __init__(self, lazy_feature: "LazyFeature"):
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
        batch_sz_bytes = batch.nbytes if isinstance(
            batch, np.ndarray) else self._lazy_feature._batch_sz_bytes
        self._lazy_feature._memory_manager.up(
            (self._lazy_feature, batch_idx, batch_sz_bytes))

        return batch


class LazyFeature:
    """
    Wraps a feature's data, allowing it to be dumped to disk and reloaded on demand.
    """

    def __init__(self, memory_manager: MemoryManager, data: list[Any], batch_size: int, path: str, batch_sz_bytes=DEFAULT_SCALAR_SIZE):
        """
        If batch_size > 1, data must be a list of np.ndarrays of shape
        [b, ...], where b = batch_size for all items except maybe the last which may be smaller.
        Path must be set and will be used for dumping when nessesary.
        """
        self._memory_manager = memory_manager
        self._data = data
        self._batch_size = batch_size
        self._path = path
        os.makedirs(path, exist_ok=True)
        self._batch_sz_bytes = batch_sz_bytes
        self._len = 0
        for i in range(len(self._data)):
            batch_sz_bytes_calc = self._data[i].nbytes if isinstance(
                self._data[i], np.ndarray) else batch_sz_bytes
            self._memory_manager.up((self, i, batch_sz_bytes_calc))
            self._len += self._data[i].shape[0]

    @staticmethod
    def from_numpy(memory_manager: MemoryManager, arr: np.ndarray, path: str) -> Self:
        # Determine the size of the batch, using array.nbytes for first element
        # and MAX_BATCH_SZ_BYTES, split data accordingly and initialize in-memory lazy feature
        batch_nbytes = arr[0].nbytes if len(arr) > 0 else MAX_BATCH_SZ_BYTES
        batch_size = max(1, 2**int(np.log2(MAX_BATCH_SZ_BYTES / batch_nbytes)))

        # Split array into batches
        batches = np.array_split(
            arr, np.arange(batch_size, arr.shape[0], batch_size), axis=0)
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

    # Use memmap here
    def dump_batch(self, i: int):
        if self._data[i] is None:
            return
        # batch_sz_bytes = self._data[i].nbytes if isinstance(
            # self._data[i], np.ndarray) else self._batch_sz_bytes
        # self._memory_manager._q.remove((self, i, batch_sz_bytes))
        # print("Dumping:", self._path, "batch:", i)
        batch_data = self._data[i]
        batch_path = os.path.join(self._path, f"batch_{i}")
        if isinstance(batch_data, np.ndarray):
            np.save(batch_path + ".npy", batch_data, allow_pickle=True)
        else:
            with open(batch_path + ".pkl", "wb") as f:
                pickle.dump(batch_data, f)
        self._data[i] = None

    def load_batch(self, i: int):
        # print("Loading:", self._path, "batch:", i)
        batch_path = os.path.join(self._path, f"batch_{i}")
        if os.path.exists(batch_path + ".npy"):
            self._data[i] = np.load(batch_path + ".npy", allow_pickle=True)
        elif os.path.exists(batch_path + ".pkl"):
            with open(batch_path + ".pkl", "rb") as f:
                self._data[i] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Batch file for index {i} not found.")

    def dump_all(self) -> None:
        for i in range(len(self._data)):
            self.dump_batch(i)

    def append_batch(self, batch: Any) -> None:
        self._data.append(batch)
        self._memory_manager.up((self, len(
            self._data) - 1, batch.nbytes if isinstance(batch, np.ndarray) else self._batch_sz_bytes))
        self._len += batch.shape[0] if isinstance(batch, np.ndarray) else 1

    def __iter__(self) -> LazyFeatureIterator:
        return LazyFeatureIterator(self)

    # Want len to be consistent with __iter__
    def __len__(self) -> int:
        return len(self._data)

    @property
    def len_samples(self) -> int:
        return self._len

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

    def __init__(self, seed: int = 42, batch_size=64, max_mem=10*1024*1024*1024, path=None, **initial_features: Any):
        """
        :param seed: Seed for controlling the random number generation (used for shuffling).
        :param initial_features: key=feature_name, value=either a single np.ndarray, a list, or scalar, etc.
        """
        self._seed = seed
        self._batch_size = batch_size
        self._memory_manager = MemoryManager(max_mem=max_mem)
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
        
        if batched:
            raise NotImplementedError(
                "Batched flat mappers aren't supported yet"
            )

        ext_type = ExtractorType.FLAT_MAPPER
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
            self.features[k] = v
            self.tables[k] = table

    def _run_mapper(self, ext: Extractor):
        """
        Run mapper extractor on dependencies stored in self.features.
        Returns outputs dictionary mapping feature names to computed arrays.
        """

        assert ext.type == ExtractorType.MAPPER or ext.type == ExtractorType.BATCH_MAPPER
        assert ext.uniform

        dep_batches = [] # Current batches of each dependency.
        dep_batch_slices = [] # Slices of current batches for each feature which we feed to mapper.
        dep_iters = [] # Iterators for each dependency.
        dep_batch_sizes = [] # Batch sizes for each dependency.
        for dep_name in ext.deps:
            if dep_name not in self.features:
                raise ValueError(
                    f"Dependency '{dep_name}' not found in self.features")
            dep_iter = iter(self.features[dep_name])
            first_batch = next(dep_iter)
            dep_batches.append(first_batch)
            dep_batch_slices.append(None)
            dep_iters.append(iter(self.features[dep_name]))
            dep_batch_sizes.append(self.features[dep_name]._batch_size)

        # Run for first sample to get output shape/type
        # TODO: do not run extractor on that sample again.
        if ext.type == ExtractorType.MAPPER:
            example_deps = [dep[0] for dep in dep_batches]
            example_outputs = ext.method(*example_deps)
            if not ext.multi_output:
                example_outputs = (np.array(example_outputs, copy=False),)
            else:
                example_outputs = tuple(np.array(x, copy=False) for x in example_outputs)
        else:
            assert ext.type == ExtractorType.BATCH_MAPPER
            example_deps = [dep[0:1] for dep in dep_batches]
            example_outputs = ext.method(*example_deps)
            if ext.multi_output:
                example_outputs = tuple(x[0] for x in example_outputs)
            else:
                example_outputs = (example_outputs[0],)

        # Calculate batch sizes based on output size
        if not isinstance(example_outputs, tuple):
            raise ValueError("Multi-output mapper must return tuple")
        output_byte_sizes = tuple(v.nbytes if isinstance(v, np.ndarray) else DEFAULT_SCALAR_SIZE for v in example_outputs)
        output_batch_sizes = tuple(
            max(1, 2**int(np.log2(MAX_BATCH_SZ_BYTES / sz))) for sz in output_byte_sizes)

        # Create output lazy features
        outputs = {}
        for i, fname in enumerate(ext.feature_names):
            outputs[fname] = LazyFeature(self._memory_manager, [], output_batch_sizes[i], os.path.join(self._path, fname))

        # batched_method must have signature (*args: np.ndarray, result: tuple[np.ndarray]) -> None
        # The outputs are written into arrays passed to result.
        if ext.type == ExtractorType.MAPPER:
            if not ext.multi_output:
                vectorized = vectorize.create_vectorized_mapper(ext.method, example_deps, example_outputs)
                def batched_method(*args: np.ndarray, result: tuple):
                    vectorized(*args, result=result[0])
            else:
                batched_method = vectorize.create_vectorized_mapper_multi(
                    ext.method, example_deps, example_outputs)
        elif ext.type == ExtractorType.BATCH_MAPPER:
            def batched_method(*args, result: tuple):
                if ext.multi_output:
                    for i, r in enumerate(ext.method(*args)):
                        np.copyto(result[i], r)
                else:
                    np.copyto(result[0], ext.method(*args))

        # Process data in batches
        n_samples = self.features[ext.deps[0]].len_samples
        min_dep_batch_size = min(dep_batch_sizes)
        min_output_batch_size = min(output_batch_sizes)
        min_batch_size = min(min_dep_batch_size, min_output_batch_size)
        output_batches = [None] * len(ext.feature_names)
        output_batch_slices = [None] * len(ext.feature_names)
        for batch_start in tqdm(range(0, n_samples, min_batch_size), total=(n_samples + min_batch_size - 1) // min_batch_size):
            # Get current batch from each dependency only when needed
            for i, dep_name in enumerate(ext.deps):
                if batch_start % dep_batch_sizes[i] == 0:
                    # Need to advance this iterator
                    dep_batches[i] = next(dep_iters[i])
                dep_batch_size = dep_batch_sizes[i]
                dep_batch_slices[i] = dep_batches[i][
                    batch_start % dep_batch_size: batch_start % dep_batch_size + min_batch_size]

            # Allocate batches if needed and store corresponding slices
            for i, fname in enumerate(ext.feature_names):
                if batch_start % output_batch_sizes[i] == 0:
                    if output_batches[i] is not None:
                        outputs[fname].append_batch(output_batches[i])
                    output_batch_size = min(output_batch_sizes[i], n_samples - batch_start)
                    output_batches[i] = np.empty(
                        (output_batch_size,) + example_outputs[i].shape, dtype=example_outputs[i].dtype)

                start_idx = batch_start % output_batch_sizes[i]
                end_idx = start_idx + min_batch_size
                output_batch_slices[i] = output_batches[i][start_idx:end_idx]

            batched_method(*dep_batch_slices, result=output_batch_slices)
            # Clean up memory after each batch
            self._memory_manager.cleanup()
        for i, f_name in enumerate(ext.feature_names):
            outputs[f_name].append_batch(output_batches[i])
        self._memory_manager.cleanup()
        return outputs

    def _run_nonuniform_mapper(self, ext: Extractor):
        """
        Run non-uniform mapper extractor on dependencies stored in self.features.
        No batching is used, as each output sample may have different size.
        Returns outputs dictionary mapping feature names to computed arrays.
        """

        assert ext.type == ExtractorType.MAPPER
        assert not ext.uniform

        dep_batches = [] # Current batches of each dependency.
        dep_batch_slices = [] # Slices of current batches for each feature which we feed to mapper.
        dep_iters = [] # Iterators for each dependency.
        dep_batch_sizes = [] # Batch sizes for each dependency.
        for dep_name in ext.deps:
            if dep_name not in self.features:
                raise ValueError(
                    f"Dependency '{dep_name}' not found in self.features")
            dep_iter = iter(self.features[dep_name])
            first_batch = next(dep_iter)
            dep_batches.append(first_batch)
            dep_batch_slices.append(None)
            dep_iters.append(iter(self.features[dep_name]))
            dep_batch_sizes.append(self.features[dep_name]._batch_size)

        # Create output lazy features
        outputs: Dict[str, LazyFeature] = {}
        for i, fname in enumerate(ext.feature_names):
            outputs[fname] = LazyFeature(self._memory_manager, [], 1, os.path.join(self._path, fname))

        # Process data in batches
        n_samples = self.features[ext.deps[0]].len_samples
        for batch_start in tqdm(range(0, n_samples), total=n_samples):
            # Get current batch from each dependency only when needed
            for i, dep_name in enumerate(ext.deps):
                if batch_start % dep_batch_sizes[i] == 0:
                    # Need to advance this iterator
                    dep_batches[i] = next(dep_iters[i])
                dep_batch_size = dep_batch_sizes[i]
                dep_batch_slices[i] = dep_batches[i][batch_start % dep_batch_size]

            result = ext.method(*dep_batch_slices)
            if not ext.multi_output:
                result = (result,)
            for i, fname in enumerate(ext.feature_names):
                outputs[fname].append_batch(np.array(result[i])[None, ...])
            # Clean up memory after each batch
            self._memory_manager.cleanup()
        return outputs      

    def _run_flatmapper(self, ext: Extractor):
        """
        Run flatmapper extractor on dependencies stored in self.features.
        No batching is used, as each output sample may have different size.
        Returns outputs dictionary mapping feature names to computed arrays.
        """

        assert ext.type == ExtractorType.FLAT_MAPPER
        assert ext.uniform

        dep_batches = [] # Current batches of each dependency.
        dep_batch_slices = [] # Slices of current batches for each feature which we feed to mapper.
        dep_iters = [] # Iterators for each dependency.
        dep_batch_sizes = [] # Batch sizes for each dependency.
        for dep_name in ext.deps:
            if dep_name not in self.features:
                raise ValueError(
                    f"Dependency '{dep_name}' not found in self.features")
            dep_iter = iter(self.features[dep_name])
            first_batch = next(dep_iter)
            dep_batches.append(first_batch)
            dep_batch_slices.append(None)
            dep_iters.append(iter(self.features[dep_name]))
            dep_batch_sizes.append(self.features[dep_name]._batch_size)

        # Run for first sample to get output shape/type
        # TODO: do not run extractor on that sample again.
        example_deps = [dep[0] for dep in dep_batches]
        example_outputs = next(iter(ext.method(*example_deps)))
        if not ext.multi_output:
            example_outputs = (np.array(example_outputs, copy=False),)
        else:
            example_outputs = tuple(np.array(x, copy=False) for x in example_outputs)

        example_outputs = example_outputs + (np.array(0),)

        # Calculate batch sizes based on output size
        if not isinstance(example_outputs, tuple):
            raise ValueError("Multi-output mapper must return tuple")
        output_byte_sizes = tuple(v.nbytes if isinstance(v, np.ndarray) else DEFAULT_SCALAR_SIZE for v in example_outputs)
        output_batch_sizes = tuple(
            max(1, 2**int(np.log2(MAX_BATCH_SZ_BYTES / sz))) for sz in output_byte_sizes)

        # Create output lazy features
        outputs = {}
        for i, fname in enumerate(ext.feature_names + [PARENT_INDEX_MAPPING_COLUMN.format(ext.feature_names[0])]):
            outputs[fname] = LazyFeature(self._memory_manager, [], output_batch_sizes[i], os.path.join(self._path, fname))

        batched_method = vectorize.create_vectorized_flatmapper(
            ext.method, example_deps, example_outputs, multi=ext.multi_output)

        # Process data in batches
        n_samples = self.features[ext.deps[0]].len_samples
        min_dep_batch_size = min(dep_batch_sizes)
        min_output_batch_size = min(output_batch_sizes)
        output_batches = [None] * (len(ext.feature_names) + 1)
        output_batch_slices = [None] * (len(ext.feature_names) + 1)
        output_batch_start = 0
        dont_create_batch = False  # hack to handle generator not writing anything
        for batch_start in tqdm(range(0, n_samples, min_dep_batch_size), 
                                total=(n_samples + min_dep_batch_size - 1) // min_dep_batch_size):
            # Get current batch from each dependency only when needed
            for i, dep_name in enumerate(ext.deps):
                if batch_start % dep_batch_sizes[i] == 0:
                    # Need to advance this iterator
                    dep_batches[i] = next(dep_iters[i])
                dep_batch_size = dep_batch_sizes[i]
                dep_batch_slices[i] = dep_batches[i][
                    batch_start % dep_batch_size: batch_start % dep_batch_size + min_dep_batch_size]

            gen = batched_method(*dep_batch_slices, base=batch_start)
            gen.send(None)

            while True:
                # Allocate batches if needed and store corresponding slices
                for i, fname in enumerate(ext.feature_names + [PARENT_INDEX_MAPPING_COLUMN.format(ext.feature_names[0])]):
                    if output_batch_start % output_batch_sizes[i] == 0 and not dont_create_batch:
                        if output_batches[i] is not None:
                            outputs[fname].append_batch(output_batches[i])
                        output_batch_size = output_batch_sizes[i]
                        output_batches[i] = np.empty(
                            (output_batch_size,) + example_outputs[i].shape, dtype=example_outputs[i].dtype)
    
                    start_idx = output_batch_start % output_batch_sizes[i]
                    end_idx = ((output_batch_start // min_output_batch_size) * min_output_batch_size) % \
                        output_batch_sizes[i]  + min_output_batch_size
                    output_batch_slices[i] = output_batches[i][start_idx:end_idx]
                
                try:
                    sz = gen.send(output_batch_slices)
                    output_batch_start += sz
                    dont_create_batch = sz == 0
                except StopIteration as e:
                    sz = e.value
                    output_batch_start += sz
                    dont_create_batch = sz == 0
                    break

            
            # Clean up memory after each batch
            self._memory_manager.cleanup()
        for i, f_name in enumerate(ext.feature_names + [PARENT_INDEX_MAPPING_COLUMN.format(ext.feature_names[0])]):
            outputs[f_name].append_batch(output_batches[i][:output_batch_start % output_batch_sizes[i]])
        self._memory_manager.cleanup()
        return outputs 

    def _run_extractor(self, ext: Extractor) -> tuple[dict[str, Any], Table]:
        # Load dependency data

        start = datetime.now()
        print(f"Running extractor for features",
              ', '.join(ext.feature_names), "at", start)
        if ext.type == ExtractorType.BATCH_MAPPER or ext.type == ExtractorType.MAPPER:
            if ext.uniform:
                results = self._run_mapper(ext)
            else:
                results = self._run_nonuniform_mapper(ext)
        elif ext.type == ExtractorType.FLAT_MAPPER:
            if ext.uniform:
                results = self._run_flatmapper(ext)
            else:
                raise NotImplementedError("Non-uniform flat mappers not supported yet.")
        else:
            raise NotImplementedError()
        end = datetime.now()
        print(f"Done running extractor", "at", end,
              "total running time", end - start)

        if ext.shuffle:
            # To shuffle elements on disk one needs to implement external memory sort, and
            # I really don't want to do it now. And probably it should be another extractor type.
            raise NotImplementedError(
                "It's hard to shuffle when elements are on disk.")
        
        index_column_name = PARENT_INDEX_MAPPING_COLUMN.format(ext.feature_names[0])
        table = self.tables[ext.deps[0]] if index_column_name not in results else Table(
            index_column_name)
        return results, table

    def _initialize_feature(self, feat_name: str, feat_value: Any) -> None:
        """
        Store the initial features in the pipeline as a LazyFeature and create corresponding tables.
        """
        lf = LazyFeature.from_numpy(
            self._memory_manager, feat_value, path=os.path.join(self._path, feat_name))
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
            # TODO: preallocate array and assign slices
            arr = np.concatenate(tuple(self.features[fn]), axis=0)
            if arr.ndim == 1:
                # shape (N,) => interpret as single column
                arr = arr.reshape(-1, 1)
            else:
                # flatten the trailing dimensions
                arr = arr.reshape(arr.shape[0], -1)
            arrays.append(arr)
        return np.hstack(arrays)

    # TODO: rewrite
    def get_columns(self, feature_names: List[str]) -> List[str]:
        """
        Return list of column names for the horizontally stacked data in `.numpy(feature_names)`.
        If a feature has shape (N, D), produce columns like: feat_0, feat_1, ...
        If shape is (N,), produce a single column "feat".
        """
        cols: List[str] = []
        for fn in feature_names:
            data = next(iter(self.features[fn]))
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
