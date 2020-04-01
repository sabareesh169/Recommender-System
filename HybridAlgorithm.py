# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:36:27 2020

@author: sabareesh
"""

from UserBased import UserBased
from ItemBased import ItemBased
from collections import defaultdict
from Metrics import Metrics
from surprise import KNNBasic

class HybridAlgorithm:
    
    def __init__(self, type_of_alg = [UserBased, ItemBased], base_alg = KNNBasic()):
        self.algorithms = []
        for algorithm in type_of_alg:
            self.algorithms.append(eval(algorithm.__name__)(base_alg, algorithm.__name__))
        
    def fit(self, ECom):
        for algorithm in self.algorithms:
            algorithm.fit(ECom)
            
    def SampleTopNRecs(self, ECom, n=10, testOrderID=68137):
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
        return {k: 1/dict1.get(k, 1000) + 1/dict2.get(k, 1000) for k in set(dict1) | set(dict2)}
    
    def Evaluate(self, ECom, n=10, verbose=True):
        metrics = {}
        print("Evaluating hit rate...")
        self.fit(ECom)
        leftOutPredictions = ECom.surpData.GetLOOCVTestSet()
        leftOutTrainingSet = ECom.surpData.GetLOOCVTrainSet()
        
        topNPredicted = defaultdict(list)
        for orderID, itemID, _ in leftOutPredictions:
            topNPredicted[int(orderID)] = self.getTopNRecs(leftOutTrainingSet, orderID)
        metrics["HR"] = Metrics.HitRate(topNPredicted, leftOutPredictions)
        return metrics