# Recommender-System

The repository contains an example of a recommender system. 

Online shopping is all over the internet. All our needs are just a click away. Recommendations systems play an impotrant part in 
satsfying the customers with personalized recommendations and can also greatly increase the profits with additional sales.

Here I take the example of an online store with orders from past two weeks. I have uploaded the part of the partial dataset due to
memory constraints.

Data dictionary:
Column title Description
- order_number Number associated with a purchase (sanitized)
- l1 Level 1 Product hierarchy (most broad)
- l2 Level 2 Product hierarchy
- l3 Level 3 Product hierarchy (most granular)
- sku Product ID (sanitized)
- brand Product brand (sanitized)

The primary objective of the project is to recommend products to customers of an online hardware store based on the products they have 
currently in the catalogue. This problem is slightly different than the standard recommender system applications as we don't track the 
user and we instead base the recommendations only on what they have in the cart now. Addditionally, there is no role for ratings as the
user has not consumed the product yet and we don't have past ratings in the dataset.

Though, most of the current recommendation methods involve using Deep Learning, I have first used collaborative filtering methods for 
this. I particularly take use of the surpriselib for the dataset framework. 

The file 'ECom.py' contains the class 'ECom' which loads the data from the txt file as a pandas dataframe and has useful methods which
can be useful for analysis. This class in turn calls the 'surpECom' which creates the surprise dataset.

For the recommendations, we can either perform user based collaborative filtering using UserBased class or item-based using ItemBased 
class with different algorithms(available in surpriselib) as the base. 

Additionally, we can use an hybrid algorithm (imported from HybridAlgorithm.py) which combines the predictions of both these algorithms. 
This has an advantage of avoiding problems like cold-start. We can also add more algorithms using the add_algorithm method to create an 
ensemble. 

All the algorithm classes have Evaluate method which reports the accuracy on leave-one-out test set.

Further, I plan to inroduce AutoEncoders, RBM and matrix factorization methods as well.
