# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:59:29 2020

@author: sabareesh
"""

from surprise import KNNBasic

import rec_modules

# Load the dataset.
ecom = rec_modules.ECom('PartialTransactions_2Weeks.txt', format='tsv')

# User based recommendations
UserKNN = KNNBasic()
userAlg = rec_modules.UserBased(UserKNN, 'user based knn')
print(userAlg.evaluate(ecom))

# Item based recommendations.
ItemKNN = KNNBasic()
itemAlg = rec_modules.ItemBased(ItemKNN, 'item based knn')
print(itemAlg.evaluate(ecom))

# Algorithm combining both the approaches.
hybrid = rec_modules.HybridAlgorithm()
print(hybrid.evaluate(ecom))
