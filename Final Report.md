
<p align="center">
<b><title>ORIE4741 Airbnb project</title></b>
</p>

<p align="center">
<img src="Pictures/head.png" width="900">
<h3>A data mining project studying Airbnb's data at Cornell University, with Dr. Madeleine Udell</h3>
</p>

### Introduction
<p>
From thousands of listings in different cities, Airbnb has become a massive sink of information. Data provided by homeowners are often big, messy, yet, extremely useful. The goal of this project is to extract knowledge from these datasets by  applying techniques and methodologies common in data mining.</p>
<p>
As more homeowners put their properties on the platform, Airbnb is able to suggest appropriate prices for the listings based on machine learning models trained over large sets of data. Our team aims to predict the prices of the listings in New York City, to allow homeowners to price their properties appropriately. Specifically, we seek to answer the following question: What prices should Airbnb suggest to their hosts given a set of features about the listing? This question is important because as more data becomes available, more intelligence can be extracted using modern machine learning tools. Therefore, it is worthwhile in exploring data driven analyses similar to those presented in this report as they are likely to improve experience for both hosts and customers, and ultimately add value to the company.
</p>


### Goal and Assumption
<p>The goal of the predictive models is clear - input the listing features X to the model, output the listing price y. However, we do not want to predict all listing prices y. We only want to to predict the price y which is reasonable and acceptable to the guest, so our model can provide the reliable price recommendation to the hosts. </p>

<p>Since we do not have the transaction records of the Airbnb listings. We assume that not all listings had successful transactions, i.e. new listings that had no customers. We further assume the listings being reviewed by guests frequently are more reasonable and acceptable to the guest. Therefore, we filtered out the listing which does not have any review score. Those listings might have unreasonable characteristics which deter the guests to rent, such as extreme high price or lack of security. We had to discard about 10,000 listings that either has no reviews or no prices. We eventually have about 28,000 listings remaining in our data set.</p>

### Feature Engineering
<p>
The Airbnb listing data we collected contains listing data from New York City from InsideAirbnb. The data file has around 38,000 listings scraped on July 3rd, 2016, and contains 100 features. Among these features are listing price, room type, amenities, and location etc. Other features include host information, room layout, amenities provided, policy, listing prices and review scores. Many of the features are non-numeric, namely, they are booleans, categorical and texts. While most features can be converted into numeric values, others will be left as booleans, categoricals, and texts and treated with appropriate regression tools. Our data set ends up having 96 features remaining.
</p>

** Amenities
<p>
The listings contain around 100 features, including numerical, and non-numerical features. For example, the amenities of a listing, i.e. TV, and WIFI, are contained in a array. Since many of the amenities are common to many properties, we began the feature transformation with the one-hot encoding method. Using this technique, we first summarized all amenities into a set of 35 unique amenities across all listings. We then expand each listing by 35 features, each representing a unique amenity initialized to zero or false. Next, we integrate through each listing and check its array of amenities. In the 35 additional columns, we set the corresponding feature to 1 or true if the same feature is found in the array. We repeat this sequence through all 28,000 plus listings, setting each entry to one to indicate an amenity. As a result, all amenities in our set are encoded by the addition of 35 new columns. We now delete the original column of arrays, leaving the one-hot encoded columns to indicate the presence of an amenity within a listing.

**Bed types
<p>
There are only 5 bed types for each listing, including Airbed, Couch, Futon, Pull-out sofa, and Real Bed. We are applying the same encoding method here, where each bed type is indicated by 1. As a result, we added five new columns of one-hot encoded features.
</p>

** Booleans
<p>
Lastly, we have many features with boolean values which indicate the characteristics of the listing and the host. For example, through these values we know whether this host is a superhost, and whether this is a instant-bookable or 24-hour check-in listing.
</p>
