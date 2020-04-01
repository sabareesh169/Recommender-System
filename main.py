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

ecom = ECom('Par Transations - 2 Weeks.txt', format='tsv')

UserKNN = KNNBasic()
userAlg = UserBased(UserKNN, 'user based knn')
userAlg.Evaluate(ecom)

ItemKNN = KNNBasic()
itemAlg = ItemBased(ItemKNN, 'item based knn')
itemAlg.Evaluate(ecom)

hybrid = HybridAlgorithm()
hybrid.Evaluate(ecom)