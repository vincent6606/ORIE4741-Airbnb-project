
<p align="center">
  <b><title>ORIE4741 Airbnb project</title></b>
</p>

<p align="center">
<h3>A data mining project studying Airbnb's data at Cornell University, with Dr. Madeleine Udell</h3>
</p>

## Introduction
<h3>Introduction</h3>
<p>
From thousands of listings in different cities, Airbnb has become a massive sink of information. Data provided by homeowners are often big, messy, yet, extremely useful. The goal of this project is to extract knowledge from these datasets by  applying techniques and methodologies common in data mining.</p>
<p>
As more homeowners put their properties on the platform, Airbnb is able to suggest appropriate prices for the listings based on machine learning models trained over large sets of data. Our team aims to predict the prices of the listings in New York City, to allow homeowners to price their properties appropriately. Specifically, we seek to answer the following question: What prices should Airbnb suggest to their hosts given a set of features about the listing? This question is important because as more data becomes available, more intelligence can be extracted using modern machine learning tools. Therefore, it is worthwhile in exploring data driven analyses similar to those presented in this report as they are likely to improve experience for both hosts and customers, and ultimately add value to the company.
</p>



## Goal and Assumption
<p>The goal of the predictive models is clear - input the listing features X to the model, output the listing price y. However, we do not want to predict all listing prices y. We only want to to predict the price y which is reasonable and acceptable to the guest, so our model can provide the reliable price recommendation to the hosts. </p>

<p>Since we do not have the transaction records of the Airbnb listings. We assume that not all listings had successful transactions, i.e. new listings that had no customers. We further assume the listings being reviewed by guests frequently are more reasonable and acceptable to the guest. Therefore, we filtered out the listing which does not have any review score. Those listings might have unreasonable characteristics which deter the guests to rent, such as extreme high price or lack of security. We had to discard about 10,000 listings that either has no reviews or no prices. We eventually have about 28,000 listings remaining in our data set.</p>
