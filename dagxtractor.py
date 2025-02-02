import inspect
import numpy as np
import pandas as pd
import os
import uuid
from collections import defaultdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Type, TypeVar
)


def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False


if is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ------------------------------------------------------------------
#  LazyFeature: holds data in memory or a path on disk
# ------------------------------------------------------------------

TData = Union[np.ndarray, List[Any]]  # Simplified data type for demonstration


class LazyFeature:
    """
    Wraps a feature's data, allowing it to be dumped to disk and reloaded on demand.
    """

    def __init__(self, data: Optional[TData] = None, path: Optional[str] = None):
        self._data = data
        self._path = path  # if data is dumped to disk
        # If _data is not None, we assume it's loaded in memory

    def is_loaded(self) -> bool:
        return self._data is not None

    def get(self) -> TData:
        """
        Return the in-memory data, loading from disk if necessary.
        """
        if self._data is None:
            if self._path is None:
                raise ValueError(
                    "LazyFeature has no data or path. Nothing to load.")
            # Load from disk
            self._data = np.load(f"{self._path}.npy", allow_pickle=True)
        return self._data

    def dump(self, path: str) -> None:
        """
        Save the current data to disk (using np.save) and clear it from memory.
        """
        if self._data is None:
            # Already unloaded or never existed
            return
        np.save(path, self._data, allow_pickle=True)
        self._data = None
        self._path = path

    def set(self, data: TData) -> None:
        """
        Overwrite the feature data in memory; if there was a path, it is still valid
        but we now consider it out of date if data changed. (For simplicity, we won't re-dump automatically.)
        """
        self._data = data

    def size(self) -> int:
        """Return number of samples (rows) if known, else 0."""
        d = self._data
        if d is None:
            # Must load to count
            d = self.get()
        if isinstance(d, np.ndarray):
            return d.shape[0]
        return len(d)


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
        cls._extractors: Dict[str, Extractor] = {}
        cls._extractors_for_feature: Dict[str, Extractor] = {}
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
        self.indexes: Dict[str, np.ndarray] = {}
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
                del self.indexes[feat]
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
        if self._has_nonempty_data(feat_name):
            return
        # If no extractor, we do nothing (it's presumably an initial feature or truly absent)
        if feat_name not in type(self)._extractors_for_feature:
            return

        ext = type(self)._extractors_for_feature[feat_name]

        # Check index references among dependencies
        dep_indexes = [self.indexes[d] for d in ext.deps]
        base_idx = dep_indexes[0]
        for idx in dep_indexes[1:]:
            if idx is not base_idx:
                raise ValueError(
                    f"Dependencies of '{feat_name}' must share the same index reference. "
                    f"Mismatch between '{ext.deps[0]}' and another dependency."
                )

        # Actually run the extractor
        outputs = self._run_extractor(ext)

        # If single-output, store under ext.feature_names[0]
        if not ext.multi_output:
            out_feat = ext.feature_names[0]
            self._assign_extracted(out_feat, outputs, ext)
        else:
            # Multi-output => 'outputs' must be dict
            if not isinstance(outputs, dict):
                raise ValueError(
                    f"Extractor for {ext.feature_names} is multi_output, but returned {type(outputs)}.")
            for fkey in ext.feature_names:
                val = outputs.get(fkey)
                if val is None:
                    raise ValueError(
                        f"Expected multi-output dict to have key '{fkey}'. Missing.")
                self._assign_extracted(fkey, val, ext)

    def _run_extractor(self, ext: Extractor) -> Any:
        """
        Actually invoke the user method. In batch mode, we pass arrays/lists in mini-batches;
        otherwise, we loop over individual samples. We apply optional shuffling if requested.

        """
        # Load dependency data
        dep_data = []
        for d in ext.deps:
            if d not in self.features:
                raise ValueError(
                    f"Dependency '{d}' not found in self.features.")
            dep_data.append(self.features[d].get())

        base_idx = self.indexes[ext.deps[0]]

        # Initialize combined_results container
        if ext.multi_output:
            # dict {feature_name: [outputs...]}
            combined_results: Any = defaultdict(list)
        else:
            combined_results = []  # single list of outputs

        # ---------------------------
        # Batch mode
        # ---------------------------
        if ext.type in (ExtractorType.BATCH_MAPPER, ExtractorType.BATCH_FLAT_MAPPER):
            # Number of mini-batches (progress bar)
            num_batches = (len(base_idx) + self._batch_size -
                           1) // self._batch_size
            # Split each dependency's data into mini-batches
            splitted_deps = []
            for dd in dep_data:
                # For arrays, np.array_split works; if dd is a list, wrap in np.array first
                arr = dd if isinstance(
                    dd, np.ndarray) else np.array(dd, dtype=object)
                splitted_deps.append(np.array_split(arr, num_batches))

            for mini_batch_idx, batch_tuple in enumerate(tqdm(zip(*splitted_deps), total=num_batches, desc=ext.feature_names[0])):
                # batch_tuple is a tuple of mini-batch arrays, one per dependency
                try:
                    r = ext.method(*batch_tuple)
                except Exception as e:
                    raise RuntimeError(
                        f"Extractor method failed on mini-batch {mini_batch_idx} "
                        f"for extractor '{ext.feature_names}'."
                    ) from e

                # Distinguish multi-output vs. single-output
                if ext.multi_output:
                    # Must be a dict of arrays
                    if not isinstance(r, dict):
                        raise TypeError(
                            f"Multi-output batch extractor must return a dict, got {type(r)} instead."
                        )
                    for k, v in r.items():
                        if not isinstance(v, np.ndarray):
                            raise TypeError(
                                f"Multi-output batch must return np.ndarray for key '{k}', got {type(v)}."
                            )
                        # BATCH_MAPPER => shape[0] should match batch size
                        if ext.type == ExtractorType.BATCH_MAPPER:
                            # first dep batch size
                            expected_sz = batch_tuple[0].shape[0]
                            if v.shape[0] != expected_sz:
                                raise ValueError(
                                    f"Expected output array for '{k}' to have shape[0] == {expected_sz}, "
                                    f"got {v.shape[0]}."
                                )
                        # Accumulate
                        combined_results[k].append(v)
                else:
                    # Single-output => r should be an array
                    if not isinstance(r, np.ndarray):
                        raise TypeError(
                            f"Batch (single-output) extractor must return an np.ndarray, got {type(r)}."
                        )
                    if ext.type == ExtractorType.BATCH_MAPPER:
                        # shape[0] must match batch size
                        expected_sz = batch_tuple[0].shape[0]
                        if r.shape[0] != expected_sz:
                            raise ValueError(
                                f"BATCH_MAPPER output shape[0] must be {expected_sz}, got {r.shape[0]}."
                            )
                    # Accumulate
                    combined_results.append(r)

        # ---------------------------
        # Per-sample mode
        # ---------------------------
        else:
            n = len(base_idx)
            for i in tqdm(range(n), total=n, desc=ext.feature_names[0]):
                # Build arguments for the i-th sample
                sample_args = []
                for dd in dep_data:
                    sample_args.append(dd[i])

                try:
                    r = ext.method(*sample_args)
                except Exception as e:
                    raise RuntimeError(
                        f"Extractor method failed on sample index {i} for '{ext.feature_names}'."
                    ) from e

                if ext.multi_output:
                    if not isinstance(r, dict):
                        raise TypeError(
                            f"Multi-output (per-sample) must return a dict, got {type(r)}."
                        )
                    for k, v in r.items():
                        if ext.type == ExtractorType.FLAT_MAPPER and ext.uniform:
                            if not isinstance(v, np.ndarray):
                                raise TypeError(
                                    f"Uniform flat mapper must return np.ndarray for key '{k}', got {type(v)}."
                                )
                        elif ext.type == ExtractorType.FLAT_MAPPER and not ext.uniform:
                            # non-uniform => convert to list
                            v = list(v)
                        combined_results[k].append(v)
                else:
                    if ext.type == ExtractorType.FLAT_MAPPER:
                        if ext.uniform:
                            if not isinstance(r, np.ndarray):
                                raise TypeError(
                                    f"Uniform flat mapper must return np.ndarray, got {type(r)}."
                                )
                        else:
                            r = list(r)
                    combined_results.append(r)

        # ---------------------------
        # Combine results
        # ---------------------------
        def _fast_concat(arrays):
            shape = (sum(array.shape[0] for array in arrays),) + arrays[0].shape[1:]
            res = np.empty(shape, dtype=arrays[0].dtype)
            start = 0
            for array in arrays:
                res[start : start + array.shape[0]] = array
                start += array.shape[0]
            return res

        def _combine_results(results: Any) -> Any:
            """
            Combine lists of arrays (or arrays) into final shape,
            depending on uniform vs. non-uniform and mapper type.
            """
            if ext.type in (ExtractorType.BATCH_MAPPER, ExtractorType.BATCH_FLAT_MAPPER):
                # We have a list of arrays -> stack along axis=0
                try:
                    return _fast_concat(results)
                except ValueError as e:
                    raise ValueError(
                        f"Could not concatenate batch results for '{ext.feature_names}'. "
                        f"Ensure consistent shapes. Error: {e}"
                    ) from e

            if ext.type == ExtractorType.FLAT_MAPPER and ext.uniform:
                # We have a list of arrays (one per sample) -> stack
                try:
                    return _fast_concat(results)
                except ValueError as e:
                    raise ValueError(
                        f"Could not concatenate uniform flat mapper results for '{ext.feature_names}'. "
                        f"Check shape consistency. Error: {e}"
                    ) from e

            if ext.type == ExtractorType.MAPPER:
                if ext.uniform:
                    # Uniform single-sample -> stack adding an axis
                    try:
                        if results and isinstance(results[0], np.ndarray):
                            shape = results[0].shape
                            # Concatenate is much faster.
                            return _fast_concat(results).reshape(-1, *shape)
                        else:
                            return np.array(results)
                    except ValueError as e:
                        raise ValueError(
                            f"Could not convert results to np.array for '{ext.feature_names}'. "
                            f"Inconsistent shapes? Error: {e}"
                        ) from e
                else:
                    # non-uniform MAPPER => keep as list
                    return results

            # non-uniform FLAT_MAPPER => sum of lists
            try:
                return sum(results, [])
            except TypeError as e:
                raise TypeError(
                    f"Non-uniform flat mapper expects a list of lists, but got something else. "
                    f"Cannot flatten results for '{ext.feature_names}'. Error: {e}"
                ) from e

        if ext.multi_output:
            if not isinstance(combined_results, dict):
                raise TypeError(
                    f"Internal error: multi_output was True but combined_results is {type(combined_results)}."
                )
            final = {}
            for k, v in combined_results.items():
                try:
                    final[k] = _combine_results(v)
                except Exception as e:
                    raise ValueError(
                        f"Failed to combine multi-output results for key '{k}' "
                        f"in extractor '{ext.feature_names}'."
                    ) from e
            combined_results = final
        else:
            try:
                combined_results = _combine_results(combined_results)
            except Exception as e:
                raise ValueError(
                    f"Failed to combine results for single-output extractor '{ext.feature_names}'."
                ) from e

        # Shuffle if needed
        if ext.shuffle:
            return self._shuffle_result(combined_results)
        return combined_results

    @staticmethod
    def _shuffle_result(result: dict | list | np.ndarray, rng: np.random.Generator) -> dict | list | np.ndarray:
        """
        Shuffle the result. Might be a single array/list or a dict of arrays/lists.
        """
        if isinstance(result, dict):
            # multi-output => each key is an array or list
            # All must be same length
            first_key = next(iter(result))
            first_val = result[first_key]
            length = len(first_val) if isinstance(
                first_val, list) else first_val.shape[0]
            perm = rng.permutation(length)

            out_dict = {}
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
                        "Unsupported type for multi-output shuffle.")
            return out_dict
        else:
            rng.shuffle(result)
            return result

    def _assign_extracted(self, feat_name: str, result: Any, ext: Extractor) -> None:
        """
        Store raw_result in self.features[feat_name] (within a LazyFeature).
        Also handle index creation/sharing logic.
        """
        dep_idx = self.indexes[ext.deps[0]] if ext.deps else None
        if (ext.type == ExtractorType.FLAT_MAPPER
            or ext.type == ExtractorType.BATCH_FLAT_MAPPER
                or ext.shuffle):
            # new index
            length = self._determine_length(result)
            new_idx = np.arange(length)
            self.indexes[feat_name] = new_idx
        else:
            # share the dependency's index
            if dep_idx is None:
                raise ValueError(
                    f"No dependencies for {feat_name}, but not a flat mapper? Unexpected.")
            self.indexes[feat_name] = dep_idx

        # Wrap in LazyFeature
        lf = LazyFeature(data=result)
        self.features[feat_name] = lf

    def _has_nonempty_data(self, feat_name: str) -> bool:
        """
        True if the feature is in self.features and has > 0 length in memory or on disk.
        """
        if feat_name not in self.features:
            return False
        lf = self.features[feat_name]
        return lf.size() > 0

    @staticmethod
    def _determine_length(raw_result: Any) -> int:
        """
        Figure out how many "rows" are in raw_result.
        """
        if isinstance(raw_result, np.ndarray):
            return raw_result.shape[0]
        if isinstance(raw_result, list):
            return len(raw_result)
        # single scalar => length 1
        return 1

    def _initialize_feature(self, feat_name: str, feat_value: Any) -> None:
        """
        Store the initial features in the pipeline as a LazyFeature.
        If it's a list, keep as list; if it's an array/scalar, treat as uniform if possible.
        Also create the index array accordingly.
        """
        if isinstance(feat_value, list):
            # store as-is
            lf = LazyFeature(feat_value)
            self.features[feat_name] = lf
            self.indexes[feat_name] = np.arange(len(feat_value))
        else:
            # single np.ndarray or scalar
            if isinstance(feat_value, np.ndarray) and feat_value.ndim > 0:
                # treat as uniform array
                lf = LazyFeature(feat_value)
                self.features[feat_name] = lf
                # index is shape[0], or if 1D => shape[0], else shape[0]
                self.indexes[feat_name] = np.arange(feat_value.shape[0])
            else:
                # scalar or zero-dim
                lf = LazyFeature([feat_value])
                self.features[feat_name] = lf
                self.indexes[feat_name] = np.arange(1)

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
        first_idx = self.indexes[feature_names[0]]
        for fn in feature_names[1:]:
            if self.indexes[fn] is not first_idx:
                raise ValueError(
                    "All requested features must share the same index object for `.numpy()`.")

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


__all__ = [DAGXtractor, mapper, flat_mapper]
