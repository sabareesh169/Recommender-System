# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:26:05 2020

@author: sabareesh
"""

from surprise import Dataset
from surprise import Reader

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut

class surpECom:
    """
    Takes in a pandas dataframe and converts into surprise dataset framework.
    params:
    df: pandas dataframe
    popRankings: dictionary sorted with popular items.
    """
    def __init__(self, df, popRankings):
        #Build a full training set for evaluating overall properties
        self.df = df
        self.data = self._convertToSurprise()
        self.rankings = popRankings
        
        # training set for the entire data
        self.fullTrainSet = self.data.build_full_trainset()
        # anti-test set for the entire training data
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        #Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(self.data, test_size=.25, random_state=1)
        
        #Build a "leave one out" train/test split for evaluating top-N recommenders
        #And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(self.data):
            self.LOOCVTrain = train
            self.LOOCVTest = test
            
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
            
    def GetFullTrainSet(self):
        return self.fullTrainSet
    
    def GetFullAntiTestSet(self):
        return self.fullAntiTestSet
    
    def GetAntiTestSetForUser(self, testSubject):
        # Get all the items not bought by a particular user
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(testSubject)
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset
    
    def GetLOOCVTrainSet(self):
        return self.LOOCVTrain
    
    def GetLOOCVTestSet(self):
        return self.LOOCVTest
    
    def GetLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet
    
    def _convertToSurprise(self):
        ''' converting the data'''
        reader = Reader(line_format='user item rating', skip_lines=1)
        surpData = Dataset.load_from_df(self.df, reader)
        return surpData
