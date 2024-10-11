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
            jpype.startJVM(classpath=['lucene-core-9.11.1.jar','lucene-search-engine.jar'])
            from java.util import HashMap, Map
            from org.example import LuceneSearchEngine
            
        assert metric == "ip"
        self.engine = LuceneSearchEngine()

    def fit(self, dataset):
        print("begin fit")
        ds = DATASETS[dataset]()
        
        index = nmslib.init(method='hnsw', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR)
        N_VEC_LIMIT = 100000 # batch size
        it = ds.get_dataset_iterator(N_VEC_LIMIT)
        id_start = 0
        for d in it:
            index.addDataPointBatch(d,np.arange(id_start,id_start+d.shape[0]))
            id_start += d.shape[0]
        start = time.time()
        index.createIndex(self.index_time_params, print_progress=True)
        end = time.time() 
        print('Index-time parameters', self.index_time_params)
        print('Indexing time = %f' % (end-start))
        self.index = index

    def load_index(self, dataset):
        return None

    def set_query_arguments(self, parameters):
        efS = parameters
        query_time_params = {'efSearch': efS}
        print('Setting query-time parameters', query_time_params)
        self.index.setQueryTimeParams(query_time_params) 

    def query(self, X, topK):
        # N, _ = X.shape
        nbrs = self.index.knnQueryBatch(X, k = topK, num_threads = 8)
        res = [x[0] for x in nbrs]
        self.I = np.array(res)
        
    def get_results(self):
        return self.I