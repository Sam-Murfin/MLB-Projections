import pandas as pd
import requests
import os
import time
import io
import zipfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read bbref csv file
df_bbref = pd.read_csv("baseball_reference_2016_clean.csv")
df_bbref['date'] = pd.to_datetime(df_bbref['date'])
#change full team names to abbreviations
team_name_map = {
    "Arizona Diamondbacks" : "ARI",
    "Atlanta Braves" : "ATL",
    "Baltimore Orioles" : "BAL",
    "Boston Red Sox" : "BOS",
    "Chicago Cubs" : "CHC",
    "Chicago White Sox" : "CWS",
    "Cincinnati Reds" : "CIN",
    "Cleveland Indians" : "CLE",
    "Colorado Rockies" : "COL",
    "Detroit Tigers" : "DET",
    "Miami Marlins" : "MIA",
    "Houston Astros" : "HOU",
    "Kansas City Royals" : "KC",
    "Los Angeles Angels of Anaheim" : "LAA",
    "Los Angeles Dodgers" : "LAD",
    "Milwaukee Brewers" : "MIL",
    "Minnesota Twins" : "MIN",
    "New York Mets" : "NYM",
    "New York Yankees" : "NYY",
    "Oakland Athletics" : "OAK",
    "Philadelphia Phillies" : "PHI",
    "Pittsburgh Pirates" : "PIT",
    "San Diego Padres" : "SD",
    "San Francisco Giants" : "SF",
    "Seattle Mariners" : "SEA",
    "St. Louis Cardinals" : "STL",
    "Tampa Bay Rays" : "TB",
    "Texas Rangers" : "TEX",
    "Toronto Blue Jays" : "TOR",
    "Washington Nationals" : "WAS"
}
#map abbrvs to bbref df
df_bbref['home_team'] = df_bbref['home_team'].map(team_name_map)
df_bbref['away_team'] = df_bbref['away_team'].map(team_name_map)

#creating df from retrosheet csv
#Download zip
url = "https://www.retrosheet.org/gamelogs/gl2016.zip"
response = requests.get(url)
response.raise_for_status()

#Extract zip in memory
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    file_name = z.namelist()[0]
    with z.open(file_name) as f:
        # Read only first 20 columns, ignore the rest
        df_rs = pd.read_csv(f, header=None, usecols=list(range(20))+ [101, 102,103,104])
# select relevant columns from retrosheet
columns = [
    "date", "number_of_game", "day_of_week", "visiting_team", "visiting_league",
    "visiting_team_game_number", "home_team", "home_league", "home_team_game_number",
    "visiting_score", "home_score", "length_outs", "day_night", "completion_info",
    "forfeit_info", "protest_info", "park_id", "attendance", "time_of_game_minutes",
    "visiting_line_score", "visiting_pitcher_ID", "visiting_pitcher_name", "home_pitcher_ID", "home_pitcher_name"
]

df_rs.columns = columns
#select relevant columns for dataframe
df_rs = df_rs[["date", "home_team", "visiting_team", "home_score", "visiting_score", "visiting_pitcher_name", "home_pitcher_name"]]
#correct column to date time
df_rs['date'] = pd.to_datetime(df_rs['date'], format='%Y%m%d')
df_rs['home_win'] = (df_rs['home_score'] > df_rs['visiting_score']).astype(int)
#map correct team abbreviations
retrosheet_to_standard = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "MIA": "MIA", "HOU": "HOU", "KCA": "KC", "LAN": "LAD",
    "MIL": "MIL", "MIN": "MIN",  "NYA": "NYY", "NYN": "NYM","OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SD", "SEA": "SEA","SFN": "SF",
    "SLN": "STL", "TBA": "TB", "TEX": "TEX", "TOR": "TOR","WAS": "WAS"
}

df_rs['home_team'] = df_rs['home_team'].map(retrosheet_to_standard)
df_rs['visiting_team'] = df_rs['visiting_team'].map(retrosheet_to_standard)
#map problem naming
manual_name_map = {
    "Jon Niese" : "Jonathon Niese",
    "Nathan Karns" : "Nate Karns",
    "Michael Fiers" : "Mike Fiers",
    "Vincent Velasquez" : "Vince Velasquez",
    "Tom Milone" : "Tommy Milone",
    "Chi Chi Gonzalez" : "ChiChi Gonzalez",
    "Hyun-Jin Ryu" : "HyunJin Ryu",
    "Fausto Carmona" : "Roberto Hernandez",
    "Rob Whalen" : "Robert Whalen"
}
df_rs['home_pitcher_name'] = df_rs['home_pitcher_name'].replace(manual_name_map)
df_rs['visiting_pitcher_name'] = df_rs['visiting_pitcher_name'].replace(manual_name_map)


#clean and normalize naming
def to_last_first(name):
    if pd.isna(name):
        return ""
    parts = name.strip().split()
    if len(parts) < 2:
        return name.lower()
    first, last = parts[0], " ".join(parts[1:])
    return f"{last},{first}"

def normalize(name):
    if pd.isna(name):
        return ""
    return name.lower().replace(" ", "").replace(".", "").replace("'", "")

def normalize_full(name):
    return normalize(to_last_first(name))

df_rs['home_starting_pitcher_norm'] = df_rs['home_pitcher_name'].apply(normalize_full)
df_rs['visiting_starting_pitcher_norm'] = df_rs['visiting_pitcher_name'].apply(normalize_full)






#load in pitching and people
pitching = pd.read_csv("Pitching.csv")
people = pd.read_csv("People.csv", encoding="latin1")

#filter for 2016
pitching_2016 = pitching[pitching['yearID'] == 2016]


pitching_2016 = pitching_2016.merge(
    people[['playerID', 'nameFirst', 'nameLast']],
    on='playerID',
    how='left'
)
pitching_2016['full_name'] = pitching_2016['nameLast'] + "," + pitching_2016['nameFirst']

def normalize(name):
    if pd.isna(name):
        return ""
    return name.lower().replace(" ", "").replace(".", "").replace("'", "")

pitching_2016['full_name'] = pitching_2016['full_name'].apply(normalize)

#select relevant stats
pitching_2016 = pitching_2016[['full_name', 'ERA', 'ER', 'SO', 'BB', 'HR', 'IPouts']]


pitching_2016['IP'] = pitching_2016['IPouts'] / 3
#collapse players who were traded midseason
pitching_2016 = pitching_2016.groupby('full_name', as_index=False).agg({
    'ER': 'sum',
    'SO': 'sum',
    'BB': 'sum',
    'HR': 'sum',
    'IP': 'sum'
})
pitching_2016['ERA'] = (pitching_2016['ER'] * 9) / pitching_2016['IP']

#merge home and visiting pitcher stats to retrosheet
df_rs = df_rs.merge(
    pitching_2016,
    how='left',
    left_on='home_starting_pitcher_norm',
    right_on='full_name',
    suffixes=('', '_home')
)
df_rs.rename(columns={
    'ERA': 'ERA_home',
    'SO': 'SO_home',
    'BB': 'BB_home',
    'HR': 'HR_home',
    'IP': 'IP_home'
}, inplace=True)

df_rs = df_rs.merge(
    pitching_2016,
    how='left',
    left_on='visiting_starting_pitcher_norm',
    right_on='full_name',
    suffixes=('', '_away')
)

#differentiate home and away pitching stats
df_rs.rename(columns={
    'ERA': 'ERA_away',
    'SO': 'SO_away',
    'BB': 'BB_away',
    'HR': 'HR_away',
    'IP': 'IP_away'
}, inplace=True)
#drop full_name columns
df_rs.drop(columns=['full_name', 'full_name_away'], inplace=True, errors='ignore')

df_model = df_rs.dropna(subset=['ERA_home', 'ERA_away'])
#defining the target(y) for whether home team won (1)
y = df_model['home_win']

#split data for tests
from sklearn.model_selection import train_test_split

X= df_model[[
'ERA_home', 'ERA_away',
    'SO_home', 'SO_away',
    'BB_home', 'BB_away',
    'HR_home', 'HR_away',
    'IP_home', 'IP_away',
]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit model for regression for win/loss projection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model= LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confustion Matrix:\n", confusion_matrix(y_test,y_pred))
print(y_probs)


# dataframe for coefficient effects
coef_df = pd.Series(model.coef_[0], index=X.columns)
coef_df.sort_values().plot(kind='barh', figsize=(10, 6), title="Feature Influence on Home Win Probability")
plt.xlabel("Coefficient Value (Log-Odds Impact)")
plt.tight_layout()
plt.show()

# roc curve
from sklearn.metrics import roc_curve, auc

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Home Win Classifier")
plt.legend()
plt.grid()
plt.show()

# predicted probabilites distribution
plt.hist(y_probs, bins=25, edgecolor='black')
plt.xlabel("Predicted Probability of Home Win")
plt.ylabel("Number of Games")
plt.title("Distribution of Predicted Home Win Probabilities")
plt.show()

# confustion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Away Win", "Home Win"],
            yticklabels=["Away Win", "Home Win"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



# dataframe for incorrect results
results = pd.DataFrame({
    'prob_home_win': y_probs,
    'predicted': y_pred,
    'actual': y_test.values
})

# confidence column for how far from 0.5 the prediction is
results['confidence'] = np.abs(results['prob_home_win'] - 0.5)

wrong_predictions = results[results['predicted'] != results['actual']]
wrong_predictions = wrong_predictions.sort_values(by='confidence', ascending=False)

print(wrong_predictions.sort_values("confidence", ascending=False).head(10))

# matching failed predictions back to games
failed_indices = wrong_predictions.index
failed_games = df_model.loc[failed_indices]

# Show useful columns
print(failed_games[['date', 'home_team', 'visiting_team',
                    'home_pitcher_name', 'visiting_pitcher_name',
                    'home_score', 'visiting_score']].head())
# visualize incorrect predictions
sns.histplot(wrong_predictions['prob_home_win'], bins=20, kde=True)
plt.axvline(0.5, color='red', linestyle='--')
plt.title("Predicted Probabilities for Incorrect Predictions")
plt.xlabel("Model Predicted Probability of Home Win")
plt.ylabel("Number of Games")
plt.show()





