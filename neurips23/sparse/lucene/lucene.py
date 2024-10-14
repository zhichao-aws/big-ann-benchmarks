import os
import numpy as np
from neurips23.sparse.base import BaseSparseANN
from benchmark.datasets import DATASETS, download_accelerated
import time
import jpype
import jpype.imports
from jpype.types import *

class LuceneEngine(BaseSparseANN):
    def __init__(self, metric, method_param):
        if not jpype.isJVMStarted():
            print("start jtype jvm")
            jpype.startJVM(
                jpype.getDefaultJVMPath(),
                "--add-modules=jdk.incubator.vector",
                classpath=['lucene-core-9.11.1.jar','lucene-search-engine.jar']
            )
            from java.util import HashMap, Map
            from org.example import LuceneSearchEngine
            
        assert metric == "ip"
        self.engine = LuceneSearchEngine("ram")
        self.name = "lucene"

    def fit(self, dataset):
        print("begin fit")
        from java.util import HashMap, Map
        ds = DATASETS[dataset]()
        
        N_VEC_LIMIT = 100000 # batch size
        it = ds.get_dataset_iterator(N_VEC_LIMIT)
        id_start = 0
        for d in it:
            sparse_vectors = []
            for vector in d:
                sparse_vector = dict(zip(map(str,vector.indices),vector.data))
                sparse_vectors.append(HashMap(sparse_vector))
            doc_ids = list(range(id_start, id_start + d.shape[0]))
            self.engine.ingest(JArray(Map)(sparse_vectors), JArray(JInt)(doc_ids))
            id_start += d.shape[0]
            print("process one block")

    def load_index(self, dataset):
        return None

    def set_query_arguments(self, parameters):
        return

    def query(self, X, topK):
        from java.util import HashMap, Map
        # N, _ = X.shape
        sparse_vectors = []
        for vector in X:
            sparse_vector = dict(zip(map(str,vector.indices),vector.data))
            sparse_vectors.append(HashMap(sparse_vector))
        res = self.engine.batchSearch(sparse_vectors, topK)
        # res = [x[0] for x in nbrs]
        self.I = np.array(res)
        
    def get_results(self):
        return self.I