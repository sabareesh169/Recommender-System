# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:34:46 2020

@author: sabareesh
"""

from Algorithm import Algorithm
from collections import defaultdict
from Metrics import Metrics

class UserBased(Algorithm):
    
    def __init__(self, algorithm, name):
        Algorithm.__init__(self, algorithm, name, sim_options={})
        self.algorithm.sim_options['user_based'] = True
        self.algorithm.sim_options['name'] = 'cosine'
        
    def getAllRecs(self,  trainSet, testOrderID=68137):
        testOrderInnerID = trainSet.to_inner_uid(testOrderID)
        similarityRow = self.simsMatrix[testOrderInnerID]
        
        similarUsers = []
        candidates = defaultdict(float)
        for innerID, score in enumerate(similarityRow):
            if (innerID != testOrderInnerID) and score>0.5:
                theirRatings = trainSet.ur[innerID]
                for rating in theirRatings:
                    candidates[rating[0]] += score
        return candidates            
        
    def SampleTopNRecs(self, ECom, n=10, testOrderID=68137):
        trainSet = ECom.surpData.fullTrainSet
        candidates = self.getAllRecs(trainSet, testOrderID)
        topN = self.getTopN(testOrderID, candidates, trainSet, n)
        print("\nReccomendations: ")
        for rec in topN:
            print(ECom.getItemName(int(rec[0])))
            
    def Evaluate(self, ECom, n=10, verbose=True):
        metrics = {}
        print("Evaluating hit rate...")
        self.fit(ECom)
        leftOutPredictions = ECom.surpData.GetLOOCVTestSet()
        leftOutTrainingSet = ECom.surpData.GetLOOCVTrainSet()
        topNPredicted = defaultdict(list)
        for orderID, itemID, _ in leftOutPredictions:
            candidates = self.getAllRecs(leftOutTrainingSet, orderID)
            topNPredicted[int(orderID)] = self.getTopN(orderID, candidates, leftOutTrainingSet, n)
        metrics["HR"] = Metrics.HitRate(topNPredicted, leftOutPredictions)
        return metrics