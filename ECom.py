# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:32:39 2020

@author: sabareesh
"""

import pandas as pd
from collections import defaultdict

from surpECom import surpECom

class ECom:

    itemID_to_name = {}
    name_to_itemID = {}

    def __init__(self, filename, format='txt', order_col='order_number', min_order_size=5):
        '''loads data from excel, csv, tsv, or txt file'''
        if format == 'excel':
            self.df = pd.read_excel(filename)
        elif format == 'csv':
            self.df = pd.read_csv(filename)
        elif format == 'tsv':
            self.df = pd.read_csv(filename, sep='\t')
        elif format == 'txt':
            self.df = pd.read_table(filename)
        else:
            raise ValueError('Invalid file format.  Please specify "excel", "csv", "tsv", or "txt".')
        
        self._preprocess(order_col='order_number', min_order_size=10)

        self.rankings = self.getPopularityRanks()
        
        self.surpData = surpECom(self.df, self.rankings)

    def getOrderPurchases(self, order):
        orderPurchases = []
        hitOrder = False
        for index, row in self.df.iterrows():
            orderID = int(row[0])
            if order == order_id:
                itemId = int(row[1])
                itemName = self.getItemName(itemID)
                orderPurchases.append((itemId, itemName))
                hitOrder = True
            if hitOrder and (orderID != order):
                break
        return orderPurchases

    def getPopularityRanks(self):
        purchases = defaultdict(int)
        rankings = defaultdict(int)
        for index, row in self.df.iterrows():
            itemID = int(row[1])
            purchases[itemID] += 1
        rank = 1
        for itemID, purchaseCount in sorted(purchases.items(), key=lambda x: x[1], reverse=True):
            rankings[itemID] = rank
            rank += 1
        return rankings
        
    def getItemName(self, itemID):
        if itemID in self.itemID_to_name:
            return self.itemID_to_name[itemID]
        else:
            return ""
        
    def getItemID(self, itemName):
        if itemName in self.name_to_itemID:
            return self.name_to_itemID[itemName]
        else:
            return 4569
        
    def _preprocess(self, order_col, min_order_size):
        self._drop_small_orders(order_col, min_order_size)
        self.df['itemID'] = self.df.groupby(['l3']).ngroup()
        self._makeItemNameDict()
        self._makeItemIdDict()
        self.df['rating'] = 1
        self.df = self.df[['order_number', 'itemID', 'rating']].sort_values('order_number')
    
    def _drop_small_orders(self, order_col='order_number', min_order_size=2):
        '''drop orders from self.data that have min_order_size or less unique items in basket'''
        self.df = self.df[self.df.groupby('order_number').order_number.transform(len) >= min_order_size]
        
    def _makeItemNameDict(self):
        self.itemID_to_name = dict(zip(self.df['itemID'], self.df['l3']))
    
    def _makeItemIdDict(self):
        self.name_to_itemID = dict(zip(self.df['l3'], self.df['itemID']))