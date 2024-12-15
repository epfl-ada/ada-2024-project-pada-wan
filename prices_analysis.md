# What is the optimal beer price ?
When launching a new product, the price range is the key point. Therefore after looking at the best products overall, we need to look at the price landscape.

The price data collected comes from Beerizer. A website selling beers coming from all over the world. It also contains ratings coming from Untappd, another website used for beer rating. 

After cleaning the dataset, we can explore the rating distribution and compare it with the ones coming from RateBeer and BeerAdvocate. We can a normal distribution with a mean and median around 3.75.  
<!---
PLOT:Put the distribution graph from Beerizer, probably better to put it at the beginning with a more general dataset analysis with all 3 datasets
-->

We can also compare all three websites ratings showing us if the average user is as critical on every place. After calculating the mean rating difference between Untappd and RateBeer, we find a difference of 0.27 were ratings are out of 5 on both. People on Untappd seem more kind. 
The same conclusion can be found when comparing average ratings between Untappd and BeerAdvocate, a difference of 0.32 ! 

## What is the most liked beer style ?

<!--
Comment: probably also before ?
-->

We first need to find liked type of beers to decide which one to brew. Only beer styles with at least 10 ratings are shown in order have meaningfull information. We can see the most liked beer styles :
- Freeze-Distilled Beer
- Stout
- Lambic
- Wild Ale
- IPA
<!--
PLOT: plot of the mean grade per general type of beer. Make it interactive with a simple slide down to be able to explore it all
-->
This gives us a good first idea of what people like. But beers from a same style can still have a lot of differences. Let's look if sub-categories shows us different conclusions. 
Looking at this new graph, we find the same styles as before with a lot of IPA, Stout and Lambic. One of these 3 might be interesting choices. This also shows us what exactly people like in these bewerages.

<!--
PLOT: mean grade per type of beer. Same as previous one
-->

The problem with these first results is that it shows what people like but doesn't assure us they will buy our product as the market might already be saturated. Therefore we need to innovate, for this we purpose we can remove beer styles with too many ratings as a correlation between number of rated beers and number of beers seems logical. Leaving us only with under-represented styles but still liked. In this case, we look at beer styles having between 5 and 20 ratings. 

<!--
PLOT: same as before, mean for under-represented types. Maybe being able to select the range ?
-->

## Price analysis

After having a first idea of popular beer types without being too mainstream. We need to look at price range in order to calculate our margins. Not every types sell at the same price, same with the beer origin. Let's first look at those two possible categories with all beer price distribution.

<!--

PLOT: price distribution by beer type and origin. Interactive graph with two choosable categories in order to see in more detail the data. Shows the full dataset
-->

Let's first look more deeply into the influence of the beer type on the price. When comparing the most expensive and the cheapest beer type, we can see a 4 fold price difference! 

<!--
PLOT: Most and least expensive beer type on average
-->

Now for the origin:

<!--
PLOT: map with average price for each country. might also be interesting to put average ratings, most popular type ? but should also do it with data from other datasets, on the same graph ? so before maybe better
-->





