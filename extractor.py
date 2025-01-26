# Code is written with ChatGPT and fixed by me (for example top sort was implemented wrong). Here is the query:

# Please write a data extraction pipeline for the audio task. It is a Python class. It has the class method register_extractor which receives a feature_name string and a method with keyword parameters. The constructor of the class receives initial features as kwargs. Extractors receive samples one by one and may return either a single sample (which is np.array of arbitrary shape) or a list of samples (in that case sample lists are merged). When method full_extract is called on object the graph of feature dependencies is built (nodes are extractors which depend on features named as their arguments). Then extractors are called in topological sort order filling the dictionary of features in object. Please support method soft_extract which recomputes only features that are not already computed, not to run the whole pipeline, when I added a new extractor.

# Also support stateful extractors. Class constructor receives boolean argument 'train'. Class method register_stateful_extractor receives feature_name string and either a method receiving features and a state if train is false, or returning features and a state (as first argument) is train is true. States are stored in dictionary corresponding to stateful_feature_extractors names.

# There is a method get_state which returns such a dictionary and set_state which updates it.
# There exists a method get_features which receives feature names, flattens all the samples (which are np.arrays) corresponding to feature names, joins flattened features and convert resulting list on np arrays to the np array


import inspect
import numpy as np
from collections import defaultdict, deque
from tqdm.notebook import tqdm

class AudioDataExtractionPipelineMeta(type):
    def __new__(cls, name, bases, dct):
        new = super().__new__(cls, name, bases, dct)
        # Registered extractors (stateless): feature_name -> callable
        new._stateless_extractors = {}
        # Dependencies for each stateless extractor: feature_name -> list of dependencies
        new._stateless_deps = {}
    
        # Registered extractors (stateful): feature_name -> callable
        new._stateful_extractors = {}
        # Dependencies for each stateful extractor: feature_name -> list of dependencies
        new._stateful_deps = {}
        return new

class AudioDataExtractionPipeline(metaclass=AudioDataExtractionPipelineMeta):
    """
    A pipeline for extracting audio features via both stateless and stateful extractors.
    
    1. register_extractor(feature_name, method):
        - Registers a feature extractor by name.
        - The method's keyword parameters (except 'state') define the input-feature dependencies.
        - The extractor is stateless.

    2. register_stateful_extractor(feature_name, method):
        - Registers a stateful extractor by name.
        - If self.train == True, the method signature is assumed to be:
            def method(state, **dependencies) -> (extracted, new_state)
        - If self.train == False, the method signature is assumed to be:
            def method(**dependencies, state) -> extracted
        - The method's keyword parameters (except 'state') define the input-feature dependencies.
        - The pipeline maintains per-feature states in a dictionary.

    3. full_extract():
        - Builds the dependency graph based on all extractors (stateless and stateful).
        - Does a topological sort on features.
        - Runs each extractor in sorted order, computing its feature if not already present.
          Re-computes every feature in the pipeline.

    4. soft_extract():
        - Same as full_extract, but if a feature is already computed, it is skipped.
          (i.e. does not re-run that extractor).

    5. get_state() and set_state():
        - Retrieve/update the dictionary of internal states for stateful extractors.

    6. get_features(feature_names):
        - Flattens all the samples (np.ndarray) for the given feature names and
          concatenates them into a single np.ndarray (row-wise).
    """

    def __init__(self, train=False, **initial_features):
        """
        Constructor.
        
        :param train: Boolean indicating whether the pipeline is in 'training' mode (matters for stateful extractors).
        :param initial_features: Any initial features passed in by name; each can be either a single np.ndarray or
                                a list of np.ndarray.
        """
        self.rnd = np.random.default_rng(42)
        self.train = train

        # Dictionary: feature_name -> list of samples (each sample is an np.ndarray)
        self.features = {}

        # Per-feature state for stateful extractors: feature_name -> arbitrary python object
        self._stateful_states = {}

        # Initialize with any user-provided features
        for feat_name, feat_value in initial_features.items():
            if isinstance(feat_value, list):
                # Assume user passed a list of samples
                self.features[feat_name] = feat_value
            else:
                # Single sample => store as a list of one
                self.features[feat_name] = [feat_value]

    @staticmethod
    def _get_dependencies_from_signature(method, ignore=('state',)):
        """
        Inspects the method signature to collect parameter names, ignoring certain ones (e.g. 'state').
        This helps determine which features (by name) are dependencies.
        """
        sig = inspect.signature(method)
        deps = []
        for name, param in sig.parameters.items():
            # Ignore parameters that are in 'ignore'
            if name in ignore:
                continue
            deps.append(name)
        return deps

    @classmethod
    def register_extractor(cls, feature_name, method, shuffle=False, map_labels=None):
        """
        Register a stateless extractor.

        The method's keyword parameters (minus 'state') are used as the
        dependencies (names of features it needs).
        
        :param feature_name: Name of the feature this extractor will compute.
        :param method: A callable that receives samples one by one with signature:
                       def method(**dependency_features) -> np.ndarray or List[np.ndarray].
                       The pipeline will call this once per sample, passing the necessary
                       dependency features.
        """
        cls._stateless_extractors[feature_name] = (method, shuffle, [] if map_labels is None else map_labels)
        cls._stateless_deps[feature_name] = cls._get_dependencies_from_signature(method, ignore=('state',))

    @classmethod
    def register_stateful_extractor(cls, feature_name, method, shuffle=False, map_labels=None):
        """
        Register a stateful extractor.

        If self.train == True, we assume the method has signature:
            def method(state, **dependency_features) -> (extracted, new_state)
        If self.train == False, we assume the method has signature:
            def method(**dependency_features, state) -> extracted

        The method's other parameters (besides 'state') are the dependencies.
        
        :param feature_name: Name of the feature this extractor will compute.
        :param method: Callable implementing the required stateful logic.
        """
        cls._stateful_extractors[feature_name] = (method, shuffle, [] if map_labels is None else map_labels)
        deps = cls._get_dependencies_from_signature(method, ignore=('state',))
        cls._stateful_deps[feature_name] = deps

        # Initialize the extractor's state if not present
        if feature_name not in cls._stateful_states:
            cls._stateful_states[feature_name] = None

    def get_state(self):
        """
        Return the entire dictionary of states for stateful features.
        """
        return self._stateful_states

    def set_state(self, state_dict):
        """
        Overwrite or add states for any stateful feature. Useful for restoring pipeline state.

        :param state_dict: A dictionary: feature_name -> state object.
        """
        for feat_name, s in state_dict.items():
            self._stateful_states[feat_name] = s

    def _build_dependency_graph(self):
        """
        Builds a dependency graph (adjacency list) of all features (both stateless and stateful).
        Keys are feature_names, values are the list of features they depend on.
        """
        graph = {}
        all_features = set()

        # Merge stateless + stateful definitions
        for feat_name, deps in self._stateless_deps.items():
            graph[feat_name] = deps[:]
            all_features.add(feat_name)
            all_features.update(deps)

        for feat_name, deps in self._stateful_deps.items():
            graph[feat_name] = deps[:]
            all_features.add(feat_name)
            all_features.update(deps)

        # Include any features that exist in the pipeline but not in the graph
        for feat_name in self.features.keys():
            if feat_name not in graph:
                graph[feat_name] = []

        return graph, all_features

    @staticmethod
    def _topological_sort(graph):
        """
        Perform a topological sort on the given dependency graph (adjacency list).
        graph: feature_name -> list of feature_names it depends on.

        Returns a list of feature_names in topologically sorted order.
        """
        order = []
        colors = defaultdict(int)
        for feat in graph.keys():
            AudioDataExtractionPipeline._dfs(graph, colors, order, feat)
        return order
        
    @staticmethod
    def _dfs(graph, colors, order, v):
        if colors[v] > 0:
            return
        colors[v] = 1
        for to in graph[v]:
            AudioDataExtractionPipeline._dfs(graph, colors, order, to)
        colors[v] = 2
        order.append(v)
    
    def full_extract(self):
        """
        Recompute *all* extractors in dependency order, overwriting or appending to self.features.

        - Build the dependency graph and topologically sort it.
        - For each feature in the sorted order, if an extractor is available, compute it for all
          samples, storing the result in self.features[feature_name].
          This recomputes features even if they already exist.
        """
        graph, _ = self._build_dependency_graph()
        sorted_feats = self._topological_sort(graph)

        for feat in sorted_feats:
            # If there's a stateless extractor for feat
            if feat in self._stateless_extractors:
                self._compute_stateless_feature(feat)
            # If there's a stateful extractor for feat
            elif feat in self._stateful_extractors:
                self._compute_stateful_feature(feat)
            # else: feat might be an initial feature with no extractor; do nothing

    def soft_extract(self):
        """
        Recompute only features that have not yet been computed (i.e. not in self.features or empty).
        - Build dependency graph and topologically sort.
        - For each feature, if we do NOT already have samples in self.features[feat], compute it.
        """
        graph, _ = self._build_dependency_graph()
        sorted_feats = self._topological_sort(graph)

        for feat in sorted_feats:
            # Skip if feature is already computed and non-empty
            if feat in self.features and len(self.features[feat]) > 0:
                continue

            # If there's a stateless extractor for feat
            if feat in self._stateless_extractors:
                self._compute_stateless_feature(feat)
            # If there's a stateful extractor for feat
            elif feat in self._stateful_extractors:
                self._compute_stateful_feature(feat)
            # else: it's just an initial feature or no extractor needed

    def _compute_stateless_feature(self, feat_name):
        """
        For a stateless feature, calls the registered extractor for every sample (in dependency features)
        and populates self.features[feat_name].
        """
        extractor, shuffle, map_labels = self._stateless_extractors[feat_name]
        deps = self._stateless_deps[feat_name]

        # Clear old data (recompute from scratch)
        self.features[feat_name] = []
        for l_from, l_to in map_labels:
            self.features[l_to] = []

        # Number of samples is inferred from one of its dependencies (assuming they match)
        if not deps:
            if map_labels:
                raise ValueError("Cannot bind labels to generator")
            # No dependencies, so we can't figure out how many samples. Let's assume 1 sample.
            # Or you might treat it as an error. We'll treat it as "no input => single sample".
            out = extractor()
            if isinstance(out, list):
                self.features[feat_name].extend(out)
            else:
                self.features[feat_name].append(out)
            if shuffle:
                self.rnd.shuffle(self.features)
            return

        num_samples = len(self.features[deps[0]])
        for i in tqdm(range(num_samples), desc=feat_name):
            # Gather the i-th sample of each dependency
            kw = {}
            for d in deps:
                kw[d] = self.features[d][i]

            result = extractor(**kw)
            # result can be a single np.array or a list of np.array
            if isinstance(result, list):
                self.features[feat_name].extend(result) 
                for l_from, l_to in map_labels:
                    self.features[l_to].extend([self.features[l_from][i]] * len(result))
            else:
                self.features[feat_name].append(result)
                for l_from, l_to in map_labels:
                    self.features[l_to].append(self.features[l_from][i])
        if shuffle:
            order = list(range(len(self.features[feat_name])))
            self.rnd.shuffle(order)
            self.features[feat_name] = [self.features[feat_name][i] for i in order]
            for l_from, l_to in map_labels:
                self.features[l_to] = [self.features[l_to][i] for i in order]
        

    def _compute_stateful_feature(self, feat_name):
        """
        For a stateful feature, calls the registered extractor for every sample (in dependency features).
        
        If self.train == True, method signature: method(state, **dependencies) -> (extracted, new_state)
        If self.train == False, method signature: method(**dependencies, state) -> extracted

        We store the updated state (if in training mode) in self._stateful_states[feat_name].
        """
        extractor, shuffle, map_labels = self._stateful_extractors[feat_name]
        deps = self._stateful_deps[feat_name]

        # Clear old data (recompute from scratch)
        self.features[feat_name] = []
        for l_from, l_to in map_labels:
            self.features[l_to] = []

        # Number of samples is inferred from one of its dependencies (if any exist)
        if len(deps) == 0:
            if map_labels:
                raise ValueError("Cannot bind labels to generator")
            # No dependencies => single call
            if self.train:
                out, new_state = extractor(state=self._stateful_states[feat_name])
                self._stateful_states[feat_name] = new_state
                if isinstance(out, list):
                    self.features[feat_name].extend(out)
                else:
                    self.features[feat_name].append(out)
            else:
                out = extractor(state=self._stateful_states[feat_name])
                if isinstance(out, list):
                    self.features[feat_name].extend(out)
                else:
                    self.features[feat_name].append(out)
            return

        num_samples = len(self.features[deps[0]])
        for i in tqdm(range(num_samples), desc=feat_name):
            # Gather the i-th sample of each dependency
            kw = {}
            for d in deps:
                kw[d] = self.features[d][i]

            if self.train:
                new_state, out = extractor(state=self._stateful_states[feat_name], **kw)
                self._stateful_states[feat_name] = new_state
            else:
                out = extractor(state=self._stateful_states[feat_name], **kw)

            # out can be a single np.array or a list of np.array
            if isinstance(out, list):
                self.features[feat_name].extend(out)
                for l_from, l_to in map_labels:
                    self.features[l_to].extend([self.features[l_from][i]] * len(out))
            else:
                self.features[feat_name].append(out)
                for l_from, l_to in map_labels:
                    self.features[l_to].append(self.features[l_from][i])

        if shuffle:
            order = list(range(len(self.features[feat_name])))
            self.rnd.shuffle(order)
            self.features[feat_name] = [self.features[feat_name][i] for i in order]
            for l_from, l_to in map_labels:
                self.features[l_to] = [self.features[l_to][i] for i in order]

    def get_features(self, feature_names):
        """
        Return a single np.ndarray containing the flattened samples for the requested feature names.
        
        - We flatten each sample's array to 1D.
        - We accumulate these flattened arrays (from each feature) into a list.
        - Finally, we convert them to an np.array.

        NOTE: This simple implementation just concatenates all samples from all features in a row-wise manner.
              (First by feature_names order, then in sample order within that feature.)
              
        :param feature_names: A list of feature names to retrieve.
        :return: A 2D np.ndarray of shape (N, M), where N is the total number of flattened samples
                 across the specified features, and M is the dimensionality of each flattened sample.
        """
        combined = []
        for sample in range(len(self.features[feature_names[0]])):
            flattened = []
            for feat_name in feature_names:
                flattened.append(self.features[feat_name][sample].reshape(-1))
            combined.append(np.hstack(flattened))
                
        if not combined:
            return np.array([])  # In case there's nothing at all
        return np.stack(combined, axis=0)

    def get_feature_names_vector(self, feature_names):
        flattened = []
        for feat_name in feature_names:
            shape = self.features[feat_name][0].shape
            flattened.extend([f'{feat_name}_{",".join(map(str, i))}' for i in np.ndindex(shape)])
        return np.array(flattened)
