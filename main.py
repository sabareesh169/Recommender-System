# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:59:29 2020

@author: sabareesh
"""

from surprise import KNNBasic

from ECom import ECom
from UserBased import UserBased
from ItemBased import ItemBased
from HybridAlgorithm import HybridAlgorithm

# Load the dataset.
ecom = ECom('Par Transations - 2 Weeks.txt', format='tsv')

# User based recommendations
UserKNN = KNNBasic()
userAlg = UserBased(UserKNN, 'user based knn')
userAlg.Evaluate(ecom)

# Item based recommendations.
ItemKNN = KNNBasic()
itemAlg = ItemBased(ItemKNN, 'item based knn')
itemAlg.Evaluate(ecom)

# Algorithm combining both the approaches.
hybrid = HybridAlgorithm()
hybrid.Evaluate(ecom)
