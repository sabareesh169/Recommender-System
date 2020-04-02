# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:33:40 2020

@author: sabareesh
"""

from surprise import accuracy
from collections import defaultdict

class Metrics:
    '''
    Measures the performance of the recommender system based on different metrics
    '''
    def MAE(predictions):
        ''' 
        Mean absolute error on the predictions of items. Lower the beeter.
        Not a useful statistics for this partiular problem as we track orders and not users
        '''
        return accuracy.mae(predictions)

    def RMSE(predictions):
        ''' 
        Root mean square error on the predictions of items. Lower the better.
        Not a useful statistics for this partiular problem as we track orders and not users
        '''
        return accuracy.rmse(predictions)

    def HitRate(topNPredicted, leftOutPredictions):
        '''
        Measures how often we are able to recommend a left-out rating. Higher is better.
        '''
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            orderID = leftOut[0]
            leftOutItemID = leftOut[1]
            # Is it in the predicted top N for this user?
            hit = False
            for itemID, rank in topNPredicted[int(orderID)]:
                if (int(leftOutItemID) == int(itemID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        '''
        Hit rate that takes the ranking into account. Higher is better.
        '''
        summation = 0
        total = 0
        # For each left-out rating
        for orderID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            for itemID, rank in topNPredicted[int(orderID)]:
                if (int(leftOutItemID) == itemID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total
