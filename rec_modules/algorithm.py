# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:34:23 2020

@author: sabareesh
"""

from operator import itemgetter

class Algorithm(object):
    '''
    Base class on which other algorithms are built.
    '''
    def __init__(self, algorithm, name, sim_options = {}):
        self.algorithm = algorithm
        self.name= name
    
    def fit(self, trainSet):
        '''
        Fit on the training set.
        Computes the similarity matrix for evaluations.
        '''
        print("\nUsing recommender ", self.GetName())
        self.algorithm.fit(trainSet)
        self.simsMatrix = self.algorithm.compute_similarities()
                
    def getTopN(self, testOrderID, candidates, trainSet, n):
        '''
        Gets the top N recommendations from all the possible recommendations.
        Makes sure we don't recommend items already bought.
        '''
        testOrderInnerID = trainSet.to_inner_uid(testOrderID)
        
        # Reconrds all the purchased items 
        ordered = {}
        for itemID, rating in trainSet.ur[testOrderInnerID]:
            ordered[itemID] = 1
            
        topN = []
        pos = 0
        for innerItemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not innerItemID in ordered:
                try:
                    itemID = trainSet.to_raw_iid(innerItemID)
                    pos += 1
                    topN.append([int(itemID),pos])
                    if (pos > n):
                        break
                except: pass
        return topN
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
