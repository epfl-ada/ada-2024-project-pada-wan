### URL to Datastory: https://owhenthesaints.github.io/posts/project/

## Abstract 
We are a group of students passionate about brewing, and eager to launch our first beer. We have therefore taken it upon ourselves to look at beer reviews, find the customer needs and succeed in our market entry. For this, we will first observe the global beer landscape to see how satisfied the customers are with the current market and if an entry is possible. In this case, we will analyse the favourable characteristics of the products in order to multiply our chances of a successful market entry. Once we have our product, we will develop a launch strategy containing A DAta-driven communication plan and launch schedule. All in all, we want to determine how to make the most successful newcomer beer! 
## Research Questions 
How to launch a successful beer?
Product launch:
	0. Dataset overview
	0.1 : data processing and dataset enrichment
1. What are good strategies to single out newcomers?
2. What is the makeup of the ideal startup location (add type of beer)?
3. When to launch a newcomer beer?


## Methods
### Data Preprocessing
The review data available from two beer rating websites, RatingBeer and BeerAdvocate, are in two .txt files that were used to create our initial dataset. This also enables us to keep only the useful data for the following tasks and discard the rest. The matched beer dataset is useful to evaluate user origin bias when rating as the dataset is created by keeping reviews from beers present in both websites, which have different demographics. The downside is a significant decrease in the number of reviews.
### Data Enrichment
In order to fully analyse, three other datasets have been used to enrich our analysis and be able to normalise the results. 
- The population per country in the world from [world bank group](https://data.worldbank.org/indicator/SP.POP.TOTL?end=2012&start=2008)
- The list of countries by beer consumption from [wikipedia](https://en.wikipedia.org/wiki/List_of_countries_by_beer_consumption_per_capita)
- Median income per year (after tax) from [Our World in Data](https://www.lisdatacenter.org)

### Time series data
Each review uploaded to RateBeer and BeerAdvocate has a date and location associated with it. This allows reviews to be analysed as a time series over a chosen period. This allows for the analysis of seasonal changes. In order to avoid bias, location must also be taken into account to ensure that reviews from the southern hemisphere have a reversed seasonal allocation. Once taken into account we can create predictive mapping of when reviews are more favourable and when they are less favourable. This can also be mapped with other data such as, for how long the people have been reviewing the data, and combined with other datasets, informing us on availability of beer all year round to give us deep insight into when we should bring our beer to market and does the optimal beer change over time. 
### Additional Datasets
The current datasets from Rate Beers and Beer Advocate are already quite dense and contain various information. However, we observed a significant concentration of users from the USA compared to other regions. To mitigate this bias, we considered two options. Firstly, we will take into account, for questions that are location dependent, the overrepresentation of U.S. users by mitigating it through normalisation using population data from the World Bank in our dataset. Secondly, we want to incorporate additional beer reviews from other datasets. We intend to complement our datasets, especially the appearance of our beers with this dataset. We also intend to get more data still off the internet by trying to scrape some data off websites such as beer advocate and beerizer.com to try and get more up-to-date info about some beers and to try and extract prices.

## Contributions
- Matti: Problem formulation, Newcomer Analysis, Data Story Structure and Text Production
- Anderson: Left the group for his startup
- Sylvain: Location Analysis, Data Story Structure and Text Production
- Owen: Data scraping, website setup, Plotting interactive graphs
- Tom: Linking the notebooks, Data Story Structure and Text Production


