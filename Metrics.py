# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:33:40 2020

@author: sabareesh
"""

from surprise import accuracy
from collections import defaultdict

class Metrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            orderID = leftOut[0]
            leftOutItemID = leftOut[1]
            # Is it in the predicted top 10 for this user?
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
