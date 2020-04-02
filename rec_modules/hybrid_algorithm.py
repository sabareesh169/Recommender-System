# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:36:27 2020

@author: sabareesh
"""

from .user_based import UserBased
from .item_based import ItemBased
from collections import defaultdict
from .metrics import Metrics
from surprise import KNNBasic

class HybridAlgorithm:
    '''
    Combines the user based and item based approaches.
    '''
    def __init__(self, type_of_alg = [UserBased, ItemBased], base_alg = \
                 [KNNBasic(sim_options={'user_based':True}), \
                  KNNBasic(sim_options={'user_based':False})]):
        self.algorithms = []
        for algorithm, base in zip(type_of_alg, base_alg):
            self.algorithms.append(eval(algorithm.__name__)(base, algorithm.__name__))
        
    def fit(self, trainSet):
        '''
        Calls fit method on all the algorithms
        '''
        for algorithm in self.algorithms:
            algorithm.fit(trainSet)
            
    def addAlgorithm(self, algorithm, base_alg):
        '''
        Possibility of adding other algorithms.
        '''
        self.algorithms.append(eval(algorithm.__name__)(base_alg, algorithm.__name__))
        
    def sampleTopNRecs(self, ECom, n=10, testOrderID=68137):
        '''
        Prints out the top N recommendations for a particular order.
        '''
        trainSet = ECom.surpData.fullTrainSet
        print('\nRecommendations:')
        for itemID, _ in self.getTopNRecs(trainSet, testOrderID, n=10):
            print(ECom.getItemName(itemID))
    
    def getTopNRecs(self, trainSet, testOrderID, n=10):
        sum_preds = {}
        for algorithm in self.algorithms:
            sum_preds = self.combine(sum_preds, algorithm.getAllRecs(trainSet, testOrderID))
        
        topN = []
        rank = 1
        for itemID, count in sorted(sum_preds.items(), key=lambda x: x[1], reverse=True):
            topN.append([itemID, rank])
            if rank>n: break
            rank += 1
        return topN
    
    def combine(self, dict1, dict2):
        '''
        Combines the predictions from the different algorithms.
        '''
        return {k: 1/dict1.get(k, 1000) + 1/dict2.get(k, 1000) for k in set(dict1) | set(dict2)}
    
    def evaluate(self, ECom, n=10, verbose=True):
        '''
        Measures the performance of the algorithm by testing on 'leave one out' training data.
        '''
        metrics = {}
        leftOutPredictions = ECom.surpData.GetLOOCVTestSet()
        leftOutTrainingSet = ECom.surpData.GetLOOCVTrainSet()
        self.fit(leftOutTrainingSet)
        
        topNPredicted = defaultdict(list)
        for orderID, itemID, _ in leftOutPredictions:
            topNPredicted[int(orderID)] = self.getTopNRecs(leftOutTrainingSet, orderID)
        print("Evaluating hit rate...")
        metrics["HR"] = Metrics.HitRate(topNPredicted, leftOutPredictions)
        return metrics
