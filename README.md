# MLB-Projections
MLB Win projections for the 2016 season using logistic regression and retrosheet data

As the sports betting world expands and crosses over with data science there is a multitude of questions that come along with that expansion. 
How much can the average data scientist learn from historical data and apply it to sports betting?
How do sportsbooks create fair betting lines?
The main question this project aims to answer is even more simple: Can we predict the outcome of an MLB game using only pitching data?


This project is very shallow dive into machine learning and its applications using Python and logistic regressions. The sports betting industry has evolved within the last decade or so where data scientists have started to use their expertise and tools to help the average sports fans and guide them in the right direction, most of the time for a small fee. Those data scientists look to match the models of the sportsbook as accurately as possible to try and get an edge on the bookies- thats obviously how you can make money betting on sports. This project will also analyze its model and show how confidently it predicts wins and losses, which could then coincide with sportsbooks moneylines to try with some additional data that is not included. 


This project uses multiple sources of data all merged together and analyzed using a PyCharm Python notebook. The first dataset comes from Kaggle, from the user Charlie Yaris, which is a dataset containing clean baseball-reference data from the 2016 MLB season. I chose this data because it was clean, concise, and contained all necessary fields (mainly team level outcomes) I wanted as the base of my project. I also pulled 2016 game log data from Retrosheet, a baseball database, and two specific datasets from the Lahman Baseball Database created by Sean Lahman, one containing pitching stats and one containing player name information.

Most of the code you will see will be the cleaning and merging of these three datasets, mainly the formatting of team names/abbreviations and player names as well as the selection of relevant columns from the retrosheet data (specifically the starting pitchers for each game and their respective stats). I then trained a model 

