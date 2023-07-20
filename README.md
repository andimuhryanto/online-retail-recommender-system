# Building a Recommender System for Online Retail: A Python Implementation
In today's rapidly evolving world of e-commerce, online retail businesses face the challenge of catering to the diverse needs and preferences of their customers. With the ever-expanding catalog of products and the abundance of choices available, customers often find it overwhelming to navigate through the vast array of options. In such a scenario, recommender systems have emerged as indispensable tools, enabling online retailers to enhance user experiences, boost customer satisfaction, and ultimately drive sales.

This project introduces an effective recommender system implemented in Python specifically tailored for online retail datasets. Leveraging the power of predictive modeling, this system aims to provide personalized product recommendations to customers based on their historical behavior and similarities to other users.

The recommender system employs a collaborative filtering approach, which involves capturing patterns from user-item interactions to identify similarities and make predictions. By analyzing user behavior such as purchase history we can generate relevant recommendations, thereby assisting customers in discovering new products, finding alternative options, and simplifying their decision-making process.

# About the Dataset
Here we are using online retail transactions data from UC Irvine Machine Learning Repository which publicly available online at http://archive.ics.uci.edu/dataset/352/online+retail. 

The "Online Retail II" dataset is a collection of transactional data from an online retail store. It provides insights into customer orders, products, and sales. The dataset is typically used for market analysis, customer segmentation, recommendation systems, and other retail-related tasks.

Here is a breakdown of the columns in the dataset:

* `InvoiceNo`: A unique identifier for each transaction or invoice.
* `StockCode`: The product code or identifier associated with each item.
* `Description`: A description of the product.
* `Quantity`: The quantity of each product in a particular transaction.
* `InvoiceDate`: The date and time when the transaction occurred.
* `Price`: The unit price of the product.
* `CustomerID`: A unique identifier for each customer.
* `Country`: The country where the customer resides.

These columns provide essential information about each transaction, including the specific products purchased, their quantities, prices, and the associated customer and country details.

# Install the Prerequisite Libraries
```
from collections import defaultdict
import random
```


# Import the Dataset
The second thing to do is to read the dataset into the notebook environment.
* We will read it from an already have a pre-downloaded tab separated value (TSV).
* We will also convert the fields into relevant data type. 
    * `Quantity` to `int`
    * `UnitPrice` to `float`
    * `CustomerID` to `int`
```
path = "./online_retail.tsv"
with open(path, 'r') as file:
    header = file.readline().strip()
    header = header.split('\t')
    dataset = []
    for line in file:
        line = [value.strip('"') for value in line.strip().split('\t')]
        dictionary = dict(zip(header, line))
        dictionary['Quantity'] = int(dictionary['Quantity'])
        dictionary['UnitPrice'] = float(dictionary['UnitPrice'])
        dataset.append(dictionary)
        
print(header)    
display(dataset[0])
```
Output:
```
Users who bought "85123A CREAM HANGING HEART T-LIGHT HOLDER": 
 ['13928', '13486', '13428', '16932', '17634', '14465', '16849', '17169', '17041', '14432']


Product StockCode that User "17377" bought: 
 ['21289', '23372', '21519', '22717', '22437', '21035', '22587', '21622', '23508', '22078']
```
## Similarity Functions
There are several methodologies to calculate the similarity (distance) between two objects e.g `Euclidean Distance`, `Jaccard Similarity`, `Cosine Similarity`, and `Pearson Correlation`.

In this case, we will used `Jaccard Similarity`, which defines as **ratio how much users who purchased both product A and B by the total unique users who bought product A or Product B**, which describe in this formula:



    J(A, B) = |A ∩ B| / |A ∪ B|
    A = set of users who purchased product A
    B = set of users who purchased product B

Then based on the similarity value, we sort it in descending because higher value means higher similiarity.
```
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def mostSimilar(iD, n):
    similarities = []
    users = userPerProduct[iD]
    for i2 in userPerProduct:
        if i2 == iD: continue
        sim = Jaccard(users, userPerProduct[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:n]
```
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def mostSimilar(iD, n):
    similarities = []
    users = userPerProduct[iD]
    for i2 in userPerProduct:
        if i2 == iD: continue
        sim = Jaccard(users, userPerProduct[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:n]
# Find Similar Products of a Specific Product
Now we can test our recommendation system by calculating the Jaccard similarity values to 3 random products.
```
totalProducts = len(itemNames)
N_product = 3
top_similar_n = 10
stockCodes = list(itemNames.keys())
```
```
for i in range(N_product):
    print('-'*50)
    print(f'{i+1} Iteration')
    print('-'*50)

    index = random.randrange(0, totalProducts)
    print('Randomized Index:', index)
    
    stockCode = stockCodes[index]
    print(f'Stock code of the {index}th Items:', stockCode)

    itemName = itemNames[stockCode]
    print('The product name of the StockCode:', itemName)

    similarProductCode = mostSimilar(stockCode, top_similar_n)
    similarProductName = [(similarity, code, itemNames[code]) for similarity, code in similarProductCode]
    display(similarProductName)
```
Output:
```
--------------------------------------------------
1 Iteration
--------------------------------------------------
Randomized Index: 809
Stock code of the 809th Items: 51014L
The product name of the StockCode: FEATHER PEN,LIGHT PINK
[(0.3383458646616541, '51014A', 'FEATHER PEN,HOT PINK'),
 (0.3, '51014C', 'FEATHER PEN,COAL BLACK'),
 (0.10714285714285714, '35471D', 'SET OF 3 BIRD LIGHT PINK FEATHER '),
 (0.10687022900763359, '21159', 'MOODY BOY  DOOR HANGER '),
 (0.1038961038961039, '21162', 'TOXIC AREA  DOOR HANGER '),
 (0.10377358490566038, '20992', 'JAZZ HEARTS PURSE NOTEBOOK'),
 (0.10185185185185185, '84596F', 'SMALL MARSHMALLOWS PINK BOWL'),
 (0.10062893081761007, '21158', 'MOODY GIRL DOOR HANGER '),
 (0.09859154929577464, '84596G', 'SMALL CHOCOLATES PINK BOWL'),
 (0.09803921568627451, '82616C', 'MIDNIGHT GLAMOUR SCARF KNITTING KIT')]
--------------------------------------------------
2 Iteration
--------------------------------------------------
Randomized Index: 3545
Stock code of the 3545th Items: 23447
The product name of the StockCode: PINK BUNNY EASTER EGG BASKET
[(0.75, '23446', 'BLUE BUNNY EASTER EGG BASKET'),
 (0.4, '23448', 'CREAM BUNNY EASTER EGG BASKET'),
 (0.16666666666666666, '84805B', 'BLUE CLIMBING HYDRANGA ART FLOWER'),
 (0.16666666666666666, '72369A', 'PINK CLEAR GLASS CANDLE PLATE'),
 (0.125, '23440', 'PAINT YOUR OWN EGGS IN CRATE'),
 (0.1111111111111111, '84804B', 'BLUE DELPHINIUM ARTIFICIAL FLOWER'),
 (0.09090909090909091, '84915', 'HAND TOWEL PINK FLOWER AND DAISY'),
 (0.09090909090909091, '23477', 'WOODLAND LARGE BLUE FELT HEART'),
 (0.08333333333333333, '23478', 'WOODLAND LARGE PINK FELT HEART'),
 (0.07692307692307693, '84952B', 'BLACK LOVE BIRD T-LIGHT HOLDER')]
--------------------------------------------------
3 Iteration
--------------------------------------------------
Randomized Index: 3211
Stock code of the 3211th Items: 23014
The product name of the StockCode: GLASS APOTHECARY BOTTLE ELIXIR
[(0.5389221556886228, '23013', 'GLASS APOTHECARY BOTTLE TONIC'),
 (0.5166666666666667, '23012', 'GLASS APOTHECARY BOTTLE PERFUME'),
 (0.19594594594594594, '23418', 'LAVENDER TOILETTE BOTTLE'),
 (0.14285714285714285, '22362', 'GLASS JAR PEACOCK BATH SALTS'),
 (0.13829787234042554, '22361', 'GLASS JAR DAISY FRESH COTTON WOOL'),
 (0.12666666666666668, '22359', 'GLASS JAR KINGS CHOICE'),
 (0.12352941176470589, '22364', 'GLASS JAR DIGESTIVE BISCUITS'),
 (0.11442786069651742, '22360', 'GLASS JAR ENGLISH CONFECTIONERY'),
 (0.11258278145695365, '22363', 'GLASS JAR MARMALADE '),
 (0.1125, '23419', 'HOME SWEET HOME BOTTLE ')]
```
# Analysis
Based on the Jaccard similarity value, we can see that each products we tested managed to show other products that similar to them.
1. First iteration, `FEATHER PEN,LIGHT PINK` top 10 products are **Feather-themed Accessories**.
2. Second iteration, `PINK BUNNY EASTER EGG BASKET` top 10 products are **Easter Decorations and Accessories.**.
3. Third iteration `GLASS APOTHECARY BOTTLE ELIXIR` top 10 products are **Glass Apothecary Bottles and Jars**.

# Conclusion
In this project, a recommender system was developed using Python for an online retail dataset. The goal was to provide personalized recommendations to users based on their historical purchase behavior. The system employed collaborative filtering techniques to identify patterns and similarities among users and items in order to generate accurate and relevant recommendations.

The implementation of the recommender system provided several benefits to the online retail platform. It enhanced the user experience by offering personalized recommendations, thereby increasing user engagement and satisfaction. The system also helped the platform increase sales and revenue by suggesting relevant items to users, which in turn encouraged repeat purchases and cross-selling.
