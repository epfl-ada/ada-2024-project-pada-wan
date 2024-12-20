# Market analysis for a too-good-to-fail and intensely delicious beer launch
## Abstract 
We are a group of students passionate about brewing, and eager to launch our first beer. We have therefore taken it upon ourselves to look at beer reviews, find the customer needs and succeed in our market entry. For this, we will first observe the global beer landscape to see how satisfied the customers are with the current market and if an entry is possible. In this case, we will analyse the favourable characteristics of the products in order to multiply our chances of a successful market entry. Once we have our product, we will develop a launch strategy containing A DAta-driven communication plan and launch schedule. All in all, we want to determine how to make the most successful newcomer beer!
## Research Questions 
### How to launch a successful beer?
#### Product launch:
1. How does the choice of initial launch country affect the ratings and reception of a new beer brand?
2. The growing hype around craft beers could lead to market saturation. Are newcomers welcome in the beer world?
#### Creating the perfect product:	
3. What kind of customer do we want to market to? Which beers do they tend to review, where do they come from and where should we export to next?
4. How does beer pricing affect the reviews and success of beer start-ups?
5. What impact do different beer styles (e.g., ales, stouts) and other taste characteristics have on consumer perception of startup beer brands?
#### Marketing strategy:
6. What beer naming strategy can promote product success?
7. How do seasonal trends affect beer perception and sales, and what are the optimal times of year to launch new beer products?

## Methods
### Data Preprocessing
The review data available from two beer rating websites, RatingBeer and BeerAdvocate, are in two .txt files that were used to create our initial dataset. This also enables us to keep only the useful data for the following tasks and discard the rest. The matched beer dataset is useful to evaluate user origin bias when rating as the dataset is created by keeping reviews from beers present in both websites, which have different demographics. The downside is a significant decrease in the number of reviews.
### Data Enrichment
To assess the optimal market entry variables for a new product, the data must first be grouped by the size of the breweries and the novelty of the beer product. Then for proper analysis, the beers need to be characterised, review texts can provide valuable context to further confirm the correct assignment of a beer into the correct category. The product name, the brewery it is from, date of first review, review development as well as review text can help determine beers that can be considered newcomers. Our definition of newcomers will include both entirely new breweries as well as beers. This data enrichment will also be brought by extra datasets.
### Time series data
Each review uploaded to RateBeer and BeerAdvocate has a date and location associated with it. This allows reviews to be analysed as a time series over a chosen period. This allows for the analysis of seasonal changes. In order to avoid bias, location must also be taken into account to ensure that reviews from the southern hemisphere have a reversed seasonal allocation. Once taken into account we can create predictive mapping of when reviews are more favourable and when they are less favourable. This can also be mapped with other data such as, for how long the people have been reviewing the data, and combined with other datasets, informing us on availability of beer all year round to give us deep insight into when we should bring our beer to market and does the optimal beer change over time. 
### Additional Datasets
The current datasets from Rate Beers and Beer Advocate are already quite dense and contain various information. However, we observed a significant concentration of users from the USA compared to other regions. To mitigate this bias, we considered two options. Firstly, we will take into account, for questions that are location dependent, the overrepresentation of U.S. users by mitigating it through normalisation using population data from the World Bank in our dataset. Secondly, we want to incorporate additional beer reviews from other datasets. We intend to complement our datasets, especially the appearance of our beers with this dataset. We also intend to get more data still off the internet by trying to scrape some data off websites such as beer advocate and beerizer.com to try and get more up-to-date info about some beers and to try and extract prices.

## Timeline and Organisation
01.12.2024 :
- Be done answering questions 1, 2, and 3 in three separate notebooks (Sylvain, Tom, Matti)
- Have a convincing website setup and ideally host questions 1 and 2 (Owen)
- Start to have a look at how to get more data from BeerAdvocate, beerizer (Owen)

08.12.2024:
- Start answering questions 4, 5, 7 (Tom, Sylvain, Matti)
- Start working on NLP for question 6 (Owen)
- Link the notebooks and link notebooks into results and upload findings to the website (Owen)

15.12.2024
- Be done answering questions 4, 5, 7 (Tom, Sylvain, Matti)
- Final linking of jupyter notebooks and upload to website (Owen)

20.12.2024 Clean up code (buffer period)


