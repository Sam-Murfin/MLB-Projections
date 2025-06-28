# MLB-Projections
MLB Win projections for the 2016 season using logistic regression and retrosheet data

As the sports betting world expands and crosses over with data science there is a multitude of questions that come along with that expansion. 
How much can the average data scientist learn from historical data and apply it to sports betting?
How do sportsbooks create fair betting lines?
The main question this project aims to answer is even more simple: Can we predict the outcome of an MLB game using only pitching data?


This project is very shallow dive into machine learning and its applications using Python and logistic regressions. The sports betting industry has evolved within the last decade or so where data scientists have started to use their expertise and tools to help the average sports fans and guide them in the right direction, most of the time for a small fee. Those data scientists look to match the models of the sportsbook as accurately as possible to try and get an edge on the bookies- thats obviously how you can make money betting on sports. This project will also analyze its model and show how confidently it predicts wins and losses, which could then coincide with sportsbooks moneylines with some additional data that is not included. 


This project uses multiple sources of data all merged together and analyzed using a PyCharm Python notebook. The first dataset comes from Kaggle, from the user Charlie Yaris, which is a dataset containing clean baseball-reference data from the 2016 MLB season. I chose this data because it was clean, concise, and contained all necessary fields (mainly team level outcomes) I wanted as the base of my project. I also pulled 2016 game log data from Retrosheet, a baseball database, and two specific datasets from the Lahman Baseball Database created by Sean Lahman, one containing pitching stats and one containing player name information.

Most of the code you will see will be the cleaning and merging of these three datasets, mainly the formatting of team names/abbreviations and player names as well as the selection of relevant columns from the retrosheet data (specifically the starting pitchers for each game and their respective stats). I then chose to run a logistic regression due to its simple and easily interpretable nature, and the binary classifications of my target (home_win). This model did not have to be scaled due to the features being on compatible scales initially. This initial model resulted in an accuracy score of 0.6213991769547325, meaning roughly 62% of samples were correctly classified. Since this is a binary classification (whether the home team won or lost the game) we can also run a confusion matrix to further summarize the performance of the model. The confusion correctly classified 124 True Negatives, 110 False Positives, 74 False Negatives, and 178 True Positives. I have visualized the heatmap below.

![MLB MLs Confusion Matrix](https://github.com/user-attachments/assets/e8df627c-6834-4517-adb8-cb08a19b2f92)



This reinforces our accuracy score of 62%. From these initial benchmarks we can further analyze our model and its accuracy and validity through some visuals. The first recognizes the effect of each individual feature (each pitching stat) on our target (home team winning or losing) on a log scale and how those features affect the odds of the home team winning. 

![MLB MLs Feature Influence on Home Win Probability](https://github.com/user-attachments/assets/66ce0b4f-fbed-4c3b-9674-31bd94529d4f)

We can easily see that the visiting teams starting pitchers earned run average (ERA) has the strongest effect on whether the home team wins or loses. This could not be more obvious when analyzing baseball results, if the visiting pitcher usually gives up more runs, that player is statistically likely to continue giving up runs and not give his team a chance to win the game. However we also see an inverse effect for the home teams starting pitcher, which is potentially why we dont see a higher accuracy score in our model-the effects nearly cancel each other out. 

We then can visualize the true positive and false positive rates as well as the overall performance of our binary classifier using a Receiver Operating Characteristic Curve, or ROC curve. An ROC curve measures the accuracy of the model by quantifying certain thresholds and how they affect those two rates. We use the Area Under Curve (AUC) value to show this accuracy, and our AUC value for this particular model was 0.64-the closer to 1.0 this value is, the better the model.  

![MLB MLs ROC Curve Home Win Classifier](https://github.com/user-attachments/assets/ec49b37e-6f51-4afe-bb4f-4f072d3e5459)



The predicted probabilities of the home team winning each game is a way to confirm how confident our model is. I have visualized the model's confidence with a histogram. 

![MLB MLs Distribution of Predicted Home Win Probabilites](https://github.com/user-attachments/assets/96a04616-ddcc-421a-8d7b-0058361e7b65)

This can tell us whether the model is just making 50/50 coin flips or is truly making accurate predictions. We can see from the histogram the model is making lots of predictions within the 40%-70% range. So not completely making a 50/50 prediction but still not the most confidence from our model. From this analysis we can consider something that sportsbooks have to consider every time the create or move a betting line- overconfident predictions. This can affect the sharpness of the public, which is not good for the book. I have created a separate dataframe containing each game in the original dataset and their predicted and actual results and have printed the top 10 incorrect predictions sorted by the models confidence in the home team winning along with the predicted probablity of a home win. We see the incorrectly predicted game with the most confidence the model had was 33%. The probabilites within these top 10 are sliced into above 70% and below 30% so we can see when the model has a high or low win probability and isnt significantly confident in that prediction. 

So what is the point of knowing how confident or how incorrect our model is when it comes to sports betting? For the sportsbook it could mean public betting pressure was misleading. For bettors it could mean similar overconfident games are ideal fade targets, especially if the model is routinely doing so. We could also change this model greatly if we include the neccesary factors that sportbooks must consider as well, things like weather and injuries that we did not include in our model. We can get even more finite if we included things like park factors, entire team stats, etc., however this model is specifically analyzing starting pitchers and wins/losses. In so many words, this model is very simple and has produced an average accuracy via logistic regression, however this model and its accuracy could easily be improved when considering other features such as the ones mentioned above, as well as comparing the historical betting lines from this particular season. 

