import os
import numpy as np
import nmslib
from neurips23.sparse.base import BaseSparseANN
from benchmark.datasets import DATASETS, download_accelerated
import time

class SparseNMSLIB(BaseSparseANN):
    def __init__(self, metric, method_param):
        assert metric == "ip"
        self.method_param = method_param
        self.name = "sparse_nnmslib"
        self.index_time_params = {'M': self.method_param["M"],
                                  'indexThreadQty': self.method_param["buildthreads"], 
                                  'efConstruction': self.method_param["efConstruction"], 
                                  'post' : 0}
        self.index = None

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