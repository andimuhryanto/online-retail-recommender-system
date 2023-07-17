---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.6
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
---

::: {.cell .markdown}
# Building a Recommender System for Online Retail: A Python Implementation
:::

::: {.cell .markdown}
In today\'s rapidly evolving world of e-commerce, online retail
businesses face the challenge of catering to the diverse needs and
preferences of their customers. With the ever-expanding catalog of
products and the abundance of choices available, customers often find it
overwhelming to navigate through the vast array of options. In such a
scenario, recommender systems have emerged as indispensable tools,
enabling online retailers to enhance user experiences, boost customer
satisfaction, and ultimately drive sales.

This project introduces an effective recommender system implemented in
Python specifically tailored for online retail datasets. Leveraging the
power of predictive modeling, this system aims to provide personalized
product recommendations to customers based on their historical behavior
and similarities to other users.

The recommender system employs a collaborative filtering approach, which
involves capturing patterns from user-item interactions to identify
similarities and make predictions. By analyzing user behavior such as
purchase history we can generate relevant recommendations, thereby
assisting customers in discovering new products, finding alternative
options, and simplifying their decision-making process.
:::

::: {.cell .markdown}
# About the Dataset
:::

::: {.cell .markdown}
Here we are using online retail transactions data from UC Irvine Machine
Learning Repository which publicly available online at
<http://archive.ics.uci.edu/dataset/352/online+retail>.

The \"Online Retail II\" dataset is a collection of transactional data
from an online retail store. It provides insights into customer orders,
products, and sales. The dataset is typically used for market analysis,
customer segmentation, recommendation systems, and other retail-related
tasks.

Here is a breakdown of the columns in the dataset:

-   `InvoiceNo`: A unique identifier for each transaction or invoice.
-   `StockCode`: The product code or identifier associated with each
    item.
-   `Description`: A description of the product.
-   `Quantity`: The quantity of each product in a particular
    transaction.
-   `InvoiceDate`: The date and time when the transaction occurred.
-   `Price`: The unit price of the product.
-   `CustomerID`: A unique identifier for each customer.
-   `Country`: The country where the customer resides.

These columns provide essential information about each transaction,
including the specific products purchased, their quantities, prices, and
the associated customer and country details.
:::

::: {.cell .markdown}
# Install the prerequisite libraries
:::

::: {.cell .code execution_count="1"}
``` python
from collections import defaultdict
import random
```
:::

::: {.cell .markdown}
# Import the Dataset
:::

::: {.cell .markdown}
The second thing to do is to read the dataset into the notebook
environment.

-   We will read it from an already have a pre-downloaded tab separated
    value (TSV).
-   We will also convert the fields into relevant data type.
    -   `Quantity` to `int`
    -   `UnitPrice` to `float`
    -   `CustomerID` to `int`
:::

::: {.cell .code execution_count="2"}
``` python
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

::: {.output .stream .stdout}
    ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
:::

::: {.output .display_data}
    {'InvoiceNo': '536365',
     'StockCode': '85123A',
     'Description': 'WHITE HANGING HEART T-LIGHT HOLDER',
     'Quantity': 6,
     'InvoiceDate': '12/1/10 8:26',
     'UnitPrice': 2.55,
     'CustomerID': '17850',
     'Country': 'United Kingdom'}
:::
:::

::: {.cell .markdown}
# Clean the data

-   Removing any transactions with not relevant Product Name
    `'DISCOUNT', 'MANUAL', '', None, 'SAMPLES', 'POSTAGE', 'PADS TO MATCH ALL CUSHIONS'`.
    This type of product name is considered noise
-   Removing any transactions those don\'t have `CustomerID`
-   Removing any transactions those `InvoiceCode` starts with the letter
    \'c\', it indicates a cancellation
-   Removing any transactions those `Quantity` or `Unit Price` are below
    0
:::

::: {.cell .code execution_count="3"}
``` python
dataset = [data for data in dataset if (data['CustomerID'] != '')]
dataset = [data for data in dataset if (data['InvoiceNo'][0].upper() != 'C')]
dataset = [data for data in dataset if (data['Quantity'] > 0)]
dataset = [data for data in dataset if (data['UnitPrice'] > 0)]
dataset = [data for data in dataset if 
           (data['Description'].upper() not in ['DISCOUNT', 'MANUAL', '', None, 'SAMPLES', 'POSTAGE', 'PADS TO MATCH ALL CUSHIONS'])]
```
:::

::: {.cell .markdown}
# Extracting Common Statistics
:::

::: {.cell .code execution_count="4"}
``` python
print('Total Transactions: ', len(dataset))
print('Total Customers Transacting:', len(set([data['CustomerID'] for data in dataset])))
print('Total Products:', len(set([data['StockCode'] for data in dataset])))
print('Total Number of Operating Countries', len(set([data['Country'] for data in dataset])))
print('Highest Transactions Volume:', max([data['Quantity'] for data in dataset]))
print('Highest Transactions Valuev:', max([data['Quantity'] * data['UnitPrice'] for data in dataset]))
```

::: {.output .stream .stdout}
    Total Transactions:  396498
    Total Customers Transacting: 4335
    Total Products: 3662
    Total Number of Operating Countries 37
    Highest Transactions Volume: 80995
    Highest Transactions Valuev: 168469.6
:::
:::

::: {.cell .markdown}
# Build a Similarity-based Recommender System

Similarity-based recommender systems are somehow trying to measure
similarity between items, or similarity between users. In this case, we
estimate the similarity between items in terms of the users who have
purchased them

This is not so much a machine learning based recommender system, but
this is trying to discover common patterns among people\'s purchasing.
:::

::: {.cell .markdown}
Here we create two `defaultdict` instances `userPerProduct` and
`productsPerUser` and will populated each of them.

-   `userPerProduct`: Contains a set of `CustomerID` which buy a
    specific product represent by its `StockCode`
-   `productsPerUser`: Contains a set of all items represented by
    `StockCode` a specific `CustomerID` bought
:::

::: {.cell .markdown}
## Populate the vector of products and users
:::

::: {.cell .code execution_count="5"}
``` python
userPerProduct = defaultdict(set)
productsPerUser = defaultdict(set)

itemNames = {}

for d in dataset:
    user,item = d['CustomerID'], d['StockCode']
    userPerProduct[item].add(user)
    productsPerUser[user].add(item)
    itemNames[item] = d['Description']
```
:::

::: {.cell .code execution_count="6"}
``` python
sampleCustomerID = '17377'
sampleStockCode = '85123A'

print(f'Users who bought "{sampleStockCode} {itemNames[sampleStockCode]}": \n', userPerProduct[sampleStockCode])
print('\n')
print(f'Product StockCode that User "{sampleCustomerID}" bought: \n', productsPerUser[sampleCustomerID])
```

::: {.output .stream .stdout}
    Users who bought "85123A CREAM HANGING HEART T-LIGHT HOLDER": 
     {'18125', '17263', '13911', '16609', '16942', '14817', '17590', '18144', '13428', '14770', '16869', '15349', '17120', '14670', '13487', '13894', '14504', '14950', '13524', '14040', '15394', '14448', '13869', '14311', '12836', '13874', '15408', '17085', '13555', '14167', '15615', '15938', '13515', '14825', '14189', '17123', '17676', '17700', '14693', '15024', '13173', '15407', '17472', '17854', '16376', '15023', '13388', '14995', '14730', '15502', '13637', '17051', '15696', '14476', '14547', '16726', '14235', '13680', '15777', '12409', '18161', '15211', '15933', '16992', '15046', '14656', '16059', '16910', '15271', '13269', '14565', '13635', '14285', '18260', '13994', '15356', '14987', '17158', '15675', '18248', '16589', '17173', '17230', '18127', '16010', '12945', '16814', '18255', '16552', '13668', '16457', '13835', '17774', '15044', '16085', '13475', '14161', '14242', '14852', '15240', '13767', '14738', '16360', '15469', '14290', '14535', '16222', '16764', '14769', '17222', '15860', '15841', '16475', '15800', '13102', '14271', '18146', '13922', '15720', '17053', '15719', '12370', '17841', '13562', '15218', '14651', '16117', '15692', '13643', '17551', '12872', '17159', '16494', '13458', '16777', '16855', '13382', '17496', '16141', '17779', '14423', '17483', '15107', '16932', '17430', '17059', '14715', '16624', '12953', '14071', '17303', '13476', '15132', '18231', '15508', '16230', '16404', '13652', '18069', '16617', '17213', '17419', '15129', '16849', '18167', '13408', '14740', '17769', '12843', '16950', '12748', '14196', '16206', '16612', '18170', '14549', '13809', '16907', '12886', '16987', '16792', '14594', '15874', '14159', '14883', '13186', '18075', '14626', '17360', '15270', '16769', '16261', '16595', '14587', '12775', '16049', '13334', '14218', '18053', '13151', '16527', '14507', '13929', '14479', '15230', '13632', '17655', '16190', '17954', '16014', '13064', '15786', '17397', '15601', '17777', '18222', '16830', '15630', '15687', '17704', '14903', '17071', '16175', '14646', '13263', '17975', '18213', '16407', '17830', '18065', '17146', '16477', '17134', '12856', '14395', '16782', '17049', '14379', '16450', '12868', '14548', '17616', '14844', '14502', '14035', '14584', '16012', '14165', '12744', '15498', '16931', '13464', '15038', '16168', '12573', '17732', '14499', '16934', '18122', '14456', '14211', '12940', '17096', '14422', '16282', '14112', '15589', '15706', '18183', '17214', '15998', '14867', '16448', '14210', '13988', '13421', '15225', '16971', '17682', '16081', '16882', '14320', '15955', '18190', '16883', '18219', '13098', '16033', '16147', '15134', '16161', '13369', '15640', '17392', '14022', '16517', '15646', '14092', '14390', '15749', '15856', '17685', '14344', '16770', '16103', '13569', '14462', '14628', '14156', '15708', '15953', '13798', '14895', '13310', '13895', '13486', '16485', '15932', '14655', '14567', '13681', '13786', '18178', '16366', '13122', '13523', '14247', '16444', '17019', '13667', '14443', '14428', '16618', '17211', '17669', '16923', '13822', '16456', '13875', '17088', '17063', '15113', '15530', '13447', '16717', '16556', '15220', '16013', '16572', '12840', '13675', '15234', '14704', '18009', '17362', '17511', '18194', '14173', '17337', '17767', '16978', '13225', '17128', '16016', '16639', '14684', '16573', '15009', '17133', '15429', '15260', '17169', '16221', '14432', '13141', '18055', '18226', '16525', '17406', '15727', '14810', '12722', '13243', '18189', '16057', '16871', '13411', '16383', '12597', '14538', '13827', '14606', '17239', '14552', '18116', '16271', '17520', '15750', '14047', '17531', '12938', '13001', '15247', '17139', '13695', '14665', '15717', '13110', '16729', '17047', '15471', '17799', '15303', '14387', '14240', '14514', '16406', '13949', '14298', '16121', '13607', '15725', '16005', '14216', '16145', '13340', '15416', '14088', '13764', '18257', '15676', '17850', '12610', '15235', '17790', '13755', '14675', '14847', '16833', '18106', '17041', '13982', '14755', '15005', '12916', '15079', '15312', '15089', '16780', '16529', '16065', '14907', '18225', '16146', '14419', '14733', '15253', '14669', '16370', '14711', '17686', '16125', '13161', '14465', '12949', '15050', '12540', '13449', '14257', '15021', '14176', '15636', '14446', '13134', '15993', '16891', '15444', '16412', '15984', '16249', '17965', '14530', '15965', '14332', '15281', '17640', '14798', '17075', '16755', '17454', '18241', '12747', '12667', '12854', '18172', '12842', '18036', '17716', '15402', '15428', '16893', '17338', '15624', '14126', '17315', '14110', '14258', '13148', '16735', '17828', '13709', '17048', '17514', '15713', '14572', '13060', '15861', '13777', '12556', '17576', '12476', '13140', '16019', '18198', '14918', '14532', '16324', '16015', '14243', '18239', '15704', '15952', '16293', '14085', '13272', '12586', '15522', '17613', '15002', '16242', '14277', '17418', '14747', '14524', '13425', '18216', '15774', '14100', '16727', '12997', '14048', '18181', '16321', '13769', '16773', '16007', '14905', '13884', '14327', '14177', '16898', '13094', '14194', '13507', '13313', '16705', '14595', '17663', '16625', '13867', '14823', '12584', '12630', '16719', '13488', '17450', '16474', '12956', '12928', '16469', '15880', '13634', '16775', '14209', '16700', '16241', '14081', '13711', '15246', '14021', '15862', '17191', '13888', '18126', '16714', '13258', '16623', '13089', '16438', '17377', '15365', '17115', '17512', '17265', '16768', '16756', '16466', '14930', '18224', '14044', '13081', '15034', '15570', '16071', '15547', '14475', '15149', '15411', '16906', '12462', '13517', '13501', '14415', '14178', '18118', '14367', '17611', '16613', '14525', '12937', '18196', '13373', '17250', '13047', '17675', '12428', '15645', '13468', '17097', '15412', '18283', '13168', '17797', '13859', '17114', '17837', '17463', '12841', '15797', '15628', '15665', '13561', '16670', '16788', '14689', '17400', '15186', '17602', '15482', '14451', '16938', '14729', '15039', '13601', '15379', '15244', '13771', '17625', '17346', '14555', '13742', '18245', '17652', '15464', '12759', '14778', '16399', '14766', '12871', '14221', '14513', '17469', '17894', '17659', '16008', '13495', '14653', '13571', '12391', '13708', '17921', '17619', '14527', '13950', '17516', '12782', '13900', '13246', '14937', '16602', '16892', '15807', '14329', '15296', '13067', '14944', '13004', '15513', '17634', '16791', '14583', '13209', '17827', '14068', '14336', '18149', '13782', '13065', '17701', '13694', '15453', '17706', '17344', '16771', '15812', '17738', '17990', '18212', '16131', '17787', '17126', '14529', '14954', '14713', '14723', '12820', '16440', '13581', '14440', '13851', '15052', '15716', '15385', '15184', '15579', '14936', '17786', '12484', '15793', '17289', '13831', '15569', '14649', '16533', '15775', '17580', '13842', '13650', '16208', '14350', '14227', '14096', '13928', '16409', '13758', '18058', '13000', '15602', '14064', '14546', '15555', '16038', '17609', '17131', '15674', '15511', '17566', '16037', '13052', '17198', '15214', '16040', '14267', '12512', '17372', '13230', '13549', '14352', '14305', '13716', '15164', '15415', '15081', '13623', '14732', '16610', '18109', '16240', '14239', '13715', '14299', '16076', '14796', '17176', '15799', '15680', '16985', '14312', '16549', '16904', '13118', '14748', '16863', '13593', '15035', '17107', '17460', '13174', '12455', '14307', '17967', '14911', '17243', '16458', '13636', '17976', '16350', '16918', '16107', '16367', '17358', '14561', '16043', '14505', '14701', '13093', '15099', '13659', '13116', '15796', '15598', '13799', '14133', '14067', '15854', '14472', '16232', '17793', '14396', '17742', '15755', '15452', '15621', '16940', '16210', '14467'}


    Product StockCode that User "17377" bought: 
     {'51014A', '21506', '47585A', '20979', '85099B', '84279P', '21507', '21874', '22662', '23439', '82494L', '85099F', '22386', '21623', '22766', '23209', '21733', '21755', '23307', '22717', '47590A', '22023', '23076', '21949', '21488', '22962', '84596F', '47566B', '21915', '22557', '21588', '22979', '35809A', '22865', '23238', '22150', '22713', '22449', '21928', '23014', '21212', '23380', '21071', '22819', '22632', '23353', '22032', '23504', '23298', '20961', '21034', '21986', '22980', '23203', '85049E', '21889', '84992', '23313', '22895', '84378', '85049G', '22437', '22353', '22989', '22113', '22435', '22029', '22078', '21035', '22900', '22990', '21070', '21929', '21519', '22629', '22080', '22866', '22024', '22027', '85049H', '23378', '22587', '22659', '84247G', '22951', '23505', '22630', '23395', '21587', '21977', '22718', '47566', '21828', '84029E', '22837', '21621', '23404', '22714', '82482', '23372', '22749', '84596B', '23371', '22358', '22111', '22907', '22616', '23356', '85123A', '23681', '85049A', '23352', '23414', '21485', '22454', '22751', '21289', '22966', '21625', '22028', '84279B', '22452', '84249A', '85049D', '21947', '23370', '22031', '20727', '23092', '23581', '22963', '21479', '22998', '22451', '21914', '23502', '21754', '22633', '22025', '23005', '23351', '21508', '23396', '22750', '22530', '84991', '84380', '23582', '23355', '22984', '21622', '20978', '22804', '22030', '23508', '23349', '22712', '23350', '21509', '23551', '46000M', '21080', '22908', '47580', '22983', '47590B', '22631', '23503', '85184C', '22553', '22035', '21527', '21559', '84535B', '46000S', '23354', '84801A', '22209', '22754', '23583', '22208', '22834', '35810A', '22960', '22759', '22352', '22994', '22211', '22077', '23393'}
:::
:::

::: {.cell .markdown}
## Similarity Functions

There are several methodologies to calculate the similarity (distance)
between two objects e.g `Euclidean Distance`, `Jaccard Similarity`,
`Cosine Similarity`, and `Pearson Correlation`.

In this case, we will used `Jaccard Similarity`, which defines as
**ratio how much users who purchased both product A and B by the total
unique users who bought product A or Product B**, which describe in this
formula:

    J(A, B) = |A ∩ B| / |A ∪ B|
    A = set of users who purchased product A
    B = set of users who purchased product B

Then based on the similarity value, we sort it in descending because
higher value means higher similiarity.
:::

::: {.cell .code execution_count="7"}
``` python
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
:::

::: {.cell .markdown}
# Find Similar Products of a Specific Product

Now we can test our recommendation system by calculating the Jaccard
similarity values to 3 random products.
:::

::: {.cell .code execution_count="8"}
``` python
totalProducts = len(itemNames)
N_product = 3
top_similar_n = 10
stockCodes = list(itemNames.keys())
```
:::

::: {.cell .code execution_count="11"}
``` python
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

::: {.output .stream .stdout}
    --------------------------------------------------
    1 Iteration
    --------------------------------------------------
    Randomized Index: 809
    Stock code of the 809th Items: 51014L
    The product name of the StockCode: FEATHER PEN,LIGHT PINK
:::

::: {.output .display_data}
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
:::

::: {.output .stream .stdout}
    --------------------------------------------------
    2 Iteration
    --------------------------------------------------
    Randomized Index: 3545
    Stock code of the 3545th Items: 23447
    The product name of the StockCode: PINK BUNNY EASTER EGG BASKET
:::

::: {.output .display_data}
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
:::

::: {.output .stream .stdout}
    --------------------------------------------------
    3 Iteration
    --------------------------------------------------
    Randomized Index: 3211
    Stock code of the 3211th Items: 23014
    The product name of the StockCode: GLASS APOTHECARY BOTTLE ELIXIR
:::

::: {.output .display_data}
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
:::
:::

::: {.cell .markdown}
# Analysis

Based on the Jaccard similarity value, we can see that each products we
tested managed to show other products that similar to them.

1.  First iteration, `FEATHER PEN,LIGHT PINK` top 10 products are
    **Feather-themed Accessories**.
2.  Second iteration, `PINK BUNNY EASTER EGG BASKET` top 10 products are
    **Easter Decorations and Accessories.**.
3.  Third iteration `GLASS APOTHECARY BOTTLE ELIXIR` top 10 products are
    **CGlass Apothecary Bottles and Jars**.
:::

::: {.cell .markdown}
# Conclusion
:::

::: {.cell .markdown}
In this project, a recommender system was developed using Python for an
online retail dataset. The goal was to provide personalized
recommendations to users based on their historical purchase behavior.
The system employed collaborative filtering techniques to identify
patterns and similarities among users and items in order to generate
accurate and relevant recommendations.

The implementation of the recommender system provided several benefits
to the online retail platform. It enhanced the user experience by
offering personalized recommendations, thereby increasing user
engagement and satisfaction. The system also helped the platform
increase sales and revenue by suggesting relevant items to users, which
in turn encouraged repeat purchases and cross-selling.
:::
