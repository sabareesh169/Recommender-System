# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:35:43 2020

@author: sabareesh
"""
from Algorithm import Algorithm
from collections import defaultdict
from Metrics import Metrics

class ItemBased(Algorithm):
    
    def __init__(self, algorithm, name):
        Algorithm.__init__(self, algorithm, name)
        self.algorithm.sim_options['user_based'] = False
        self.algorithm.sim_options['name'] = 'cosine'
            
    def getAllRecs(self,  trainSet, testOrderID=68137):
        testOrderInnerID = trainSet.to_inner_uid(testOrderID)
        orderedItems = trainSet.ur[testOrderInnerID]

        candidates = defaultdict(float)
        for innerItemID, rating in orderedItems:
            similarityRow = self.simsMatrix[innerItemID]
            for innerID, score in enumerate(similarityRow):
                if score>0.5:
                    candidates[innerID] += score 
        return candidates            
        
    def SampleTopNRecs(self, ECom, n=10, testOrderID=68137):
        trainSet = ECom.surpData.fullTrainSet
        candidates = self.getAllRecs(trainSet, testOrderID)
        topN = Algorithm.getTopN(self, testOrderID, candidates, trainSet, n)
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