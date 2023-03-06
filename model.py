import numpy as np
import pandas as pd

import webdriver_manager
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time

import requests
from bs4 import BeautifulSoup
import re
import json

from datetime import date

import seaborn as sns
import matplotlib.pyplot as plt

# ----- Global variables ------------------------------------------------------------
curr_games_played = [80]
past_games_played = [10, 40, 80]

today = str(date.today())

# WebDriver Options
opts = Options()
opts.add_argument("--headless")
opts.add_experimental_option("excludeSwitches", ["enable-automation"])
opts.add_experimental_option('useAutomationExtension', False)



# =============================================== [--- Downloading data ---] ===============================================
def downloadData(fromSeason, thruSeason, today, games_played):
    global curr_data_home_GF, curr_data_home_GA, curr_data_away_GF, curr_data_away_GA
    print("Downloading data...")
    curr_data_home = []
    curr_data_away = []
    counter = 1
    
    for gps in games_played:
        # urls to access current data
        urlAway = f"https://www.naturalstattrick.com/teamtable.php?fromseason={fromSeason}&thruseason={thruSeason}&stype=2&sit=5v5&score=all&rate=y&team=all&loc=A&gpf=c&gp={gps}&fd=&td={today}"
        urlHome = f"https://www.naturalstattrick.com/teamtable.php?fromseason={fromSeason}&thruseason={thruSeason}&stype=2&sit=5v5&score=all&rate=y&team=all&loc=H&gpf=c&gp={gps}&fd=&td={today}"

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = opts) # downloads and sets newest chromedriver
        params = {'behavior': 'allow', 'downloadPath': r'currentData'}
        driver.execute_cdp_cmd('Page.setDownloadBehavior', params) # download behaviour set to this directory

        driver.get(urlAway) # driver launches
        
        filterButton = driver.find_element(By.ID, 'colfilterlb')
        filterButton.click()
        buttons = driver.find_elements(By.CSS_SELECTOR, '[class*="\\\\buttonspan"]')
        button_order = [7, 8, 10, 12, 13, 14, 15, 16, 17]

        for i in button_order:
            buttons[i].click()

        saveButton = driver.find_elements(By.TAG_NAME, "input")[29] # save button
        saveButton.click()
        time.sleep(3)
        
        curr_data_away.append(pd.read_csv('currentData/games.csv', usecols = ["Team","GP","TOI/GP","W","L","OTL","ROW","Points","Point %","CF/60","CA/60","CF%","FF/60","FA/60","FF%","SF/60","SA/60","SF%","GF/60","GA/60","GF%","xGF/60","xGA/60","xGF%","SCF/60","SCA/60","SCF%","SCSF/60","SCSA/60","SCSF%","SCGF/60","SCGA/60","SCGF%","SCSH%","SCSV%","HDCF/60","HDCA/60","HDCF%","HDSF/60","HDSA/60","HDSF%","HDGF/60","HDGA/60","HDGF%","HDSH%","HDSV%","MDCF/60","MDCA/60","MDCF%","MDSF/60","MDSA/60","MDSF%","MDGF/60","MDGA/60","MDGF%","MDSH%","MDSV%","LDCF/60","LDCA/60","LDCF%","LDSF/60","LDSA/60","LDSF%","LDGF/60","LDGA/60","LDGF%","LDSH%","LDSV%","SH%","SV%","PDO"], header = 0))
        
        print("(" + str(counter) +  ")" + "done away: " + str(gps) + " games played.") # i.e. (1) done away: 10 games played
        
        driver.close()
        driver.quit()
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = opts) # downloads and sets newest chromedriver
        params = {'behavior': 'allow', 'downloadPath': r'currentData'}
        driver.execute_cdp_cmd('Page.setDownloadBehavior', params) # download behaviour set to this directory
        
        driver.get(urlHome)
        
        filterButton = driver.find_element(By.ID, 'colfilterlb')
        filterButton.click()
        buttons = driver.find_elements(By.CSS_SELECTOR, '[class*="\\\\buttonspan"]')

        for i in button_order:
            buttons[i].click()

        saveButton = driver.find_elements(By.TAG_NAME, "input")[29] # save button
        saveButton.click()
        time.sleep(3)
        
        curr_data_home.append(pd.read_csv('currentData/games.csv', usecols = ["Team","GP","TOI/GP","W","L","OTL","ROW","Points","Point %","CF/60","CA/60","CF%","FF/60","FA/60","FF%","SF/60","SA/60","SF%","GF/60","GA/60","GF%","xGF/60","xGA/60","xGF%","SCF/60","SCA/60","SCF%","SCSF/60","SCSA/60","SCSF%","SCGF/60","SCGA/60","SCGF%","SCSH%","SCSV%","HDCF/60","HDCA/60","HDCF%","HDSF/60","HDSA/60","HDSF%","HDGF/60","HDGA/60","HDGF%","HDSH%","HDSV%","MDCF/60","MDCA/60","MDCF%","MDSF/60","MDSA/60","MDSF%","MDGF/60","MDGA/60","MDGF%","MDSH%","MDSV%","LDCF/60","LDCA/60","LDCF%","LDSF/60","LDSA/60","LDSF%","LDGF/60","LDGA/60","LDGF%","LDSH%","LDSV%","SH%","SV%","PDO"], header = 0))

        print("(" + str(counter) +  ")" + "done home: " + str(gps) + " games played.") # i.e. (2) done home: 10 games played
        counter = counter + 1
        
        driver.close()
        driver.quit()
    
    # return home data, and away data
    return curr_data_home, curr_data_away

def get_elo():
    url = 'https://projects.fivethirtyeight.com/2023-nhl-predictions/'
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    
    # get the table (id = standings-table)
    table = soup.find('table', attrs={'id':'standings-table'})
    
    # go into the table body
    table_body = table.find('tbody')
    
    # every row in the table, get the team name and the spi rating
    rows = table_body.find_all('tr')
    elo = {}
    for row in rows:
        # remove the number at the end of the team name (there is no space between the name and the number)
        elo[re.split('(\d+)', row.find('td', attrs={'class':'name'}).text)[0]] = row.find('td', attrs={'class':'elo'}).text
        
    # elo rating are currently between 1400 and 1700 change it to be between 0 and 1
    for key in elo:
        elo[key] = (float(elo[key]) - 1300) / 300
    
    return elo

def downloadCurrentData():
    curr_home_data, curr_away_data = downloadData(20222023, 20222023, today, curr_games_played)
    
    # convert to dataframes
    curr_home_data_df = pd.concat(curr_home_data)
    curr_away_data_df = pd.concat(curr_away_data)
    
    # save to csv
    curr_home_data_df.to_csv('currentData/home.csv', index = False)
    curr_away_data_df.to_csv('currentData/away.csv', index = False)

def downloadPastData():
    seasons = [
        [20212022, 20222023],
        [20192020, 20202021],
        [20182019, 20192020],
        [20172018, 20182019],
        [20162017, 20172018],
        [20152016, 20162017],
        [20142015, 20152016],
        [20132014, 20142015],
        [20122013, 20132014],
        [20112012, 20122013],
        [20102011, 20112012],
        [20092010, 20102011],
    ]
    
    all_away_data = []
    all_home_data = []
    
    for season in seasons:
        print("Downloading data for season: " + str(season[0]) + "-" + str(season[1]))
        home_data, away_data = downloadData(season[0], season[1], today, past_games_played)
        
        # convert to dataframes
        home_data_df = pd.concat(home_data)
        away_data_df = pd.concat(away_data)
        
        # append to list of dataframes
        all_away_data.append(away_data_df)
        all_home_data.append(home_data_df)
        
    # concatenate all dataframes
    all_away_df = pd.concat(all_away_data)
    all_home_df = pd.concat(all_home_data)
    
    print("Saving data to csv")
    all_away_df.to_csv('pastData/all_away_data.csv', index = False)
    all_home_df.to_csv('pastData/all_home_data.csv', index = False)
    
def add_elo(df, elo):
    # convert elo values to floats
    elo = {key: float(value) for key, value in elo.items()}
    df['ELO'] = df.index.map(lambda x: elo[get_team_from_team_name(elo, x)])
    return df

def get_team_from_team_name(elo, team):
    for key in elo:
        if key in team:
            return key
        
    return "Devils"

#downloadPastData() # if you want to download past data again
#downloadCurrentData() # if you want to download current data again

# ---------- Get Data ----------

current_home_df = pd.read_csv('currentData/home.csv', usecols = ["Team","GP","TOI/GP","W","L","OTL","ROW","Points","Point %","CF/60","CA/60","CF%","FF/60","FA/60","FF%","SF/60","SA/60","SF%","GF/60","GA/60","GF%","xGF/60","xGA/60","xGF%","SCF/60","SCA/60","SCF%","SCSF/60","SCSA/60","SCSF%","SCGF/60","SCGA/60","SCGF%","SCSH%","SCSV%","HDCF/60","HDCA/60","HDCF%","HDSF/60","HDSA/60","HDSF%","HDGF/60","HDGA/60","HDGF%","HDSH%","HDSV%","MDCF/60","MDCA/60","MDCF%","MDSF/60","MDSA/60","MDSF%","MDGF/60","MDGA/60","MDGF%","MDSH%","MDSV%","LDCF/60","LDCA/60","LDCF%","LDSF/60","LDSA/60","LDSF%","LDGF/60","LDGA/60","LDGF%","LDSH%","LDSV%","SH%","SV%","PDO"], header = 0)
current_away_df = pd.read_csv('currentData/away.csv', usecols = ["Team","GP","TOI/GP","W","L","OTL","ROW","Points","Point %","CF/60","CA/60","CF%","FF/60","FA/60","FF%","SF/60","SA/60","SF%","GF/60","GA/60","GF%","xGF/60","xGA/60","xGF%","SCF/60","SCA/60","SCF%","SCSF/60","SCSA/60","SCSF%","SCGF/60","SCGA/60","SCGF%","SCSH%","SCSV%","HDCF/60","HDCA/60","HDCF%","HDSF/60","HDSA/60","HDSF%","HDGF/60","HDGA/60","HDGF%","HDSH%","HDSV%","MDCF/60","MDCA/60","MDCF%","MDSF/60","MDSA/60","MDSF%","MDGF/60","MDGA/60","MDGF%","MDSH%","MDSV%","LDCF/60","LDCA/60","LDCF%","LDSF/60","LDSA/60","LDSF%","LDGF/60","LDGA/60","LDGF%","LDSH%","LDSV%","SH%","SV%","PDO"], header = 0)

past_home_df = pd.read_csv('pastData/all_home_data.csv', usecols = ["Team","GP","TOI/GP","W","L","OTL","ROW","Points","Point %","CF/60","CA/60","CF%","FF/60","FA/60","FF%","SF/60","SA/60","SF%","GF/60","GA/60","GF%","xGF/60","xGA/60","xGF%","SCF/60","SCA/60","SCF%","SCSF/60","SCSA/60","SCSF%","SCGF/60","SCGA/60","SCGF%","SCSH%","SCSV%","HDCF/60","HDCA/60","HDCF%","HDSF/60","HDSA/60","HDSF%","HDGF/60","HDGA/60","HDGF%","HDSH%","HDSV%","MDCF/60","MDCA/60","MDCF%","MDSF/60","MDSA/60","MDSF%","MDGF/60","MDGA/60","MDGF%","MDSH%","MDSV%","LDCF/60","LDCA/60","LDCF%","LDSF/60","LDSA/60","LDSF%","LDGF/60","LDGA/60","LDGF%","LDSH%","LDSV%","SH%","SV%","PDO"], header = 0)
past_away_df = pd.read_csv('pastData/all_away_data.csv', usecols = ["Team","GP","TOI/GP","W","L","OTL","ROW","Points","Point %","CF/60","CA/60","CF%","FF/60","FA/60","FF%","SF/60","SA/60","SF%","GF/60","GA/60","GF%","xGF/60","xGA/60","xGF%","SCF/60","SCA/60","SCF%","SCSF/60","SCSA/60","SCSF%","SCGF/60","SCGA/60","SCGF%","SCSH%","SCSV%","HDCF/60","HDCA/60","HDCF%","HDSF/60","HDSA/60","HDSF%","HDGF/60","HDGA/60","HDGF%","HDSH%","HDSV%","MDCF/60","MDCA/60","MDCF%","MDSF/60","MDSA/60","MDSF%","MDGF/60","MDGA/60","MDGF%","MDSH%","MDSV%","LDCF/60","LDCA/60","LDCF%","LDSF/60","LDSA/60","LDSF%","LDGF/60","LDGA/60","LDGF%","LDSH%","LDSV%","SH%","SV%","PDO"], header = 0)

# =============================================== [--- Data Pre-Processing  ---] ===============================================

# normalize data
def normalizeData(df):
    df = df.drop(columns = ['GP']) # drop team name and GP column
    
    # replace any - with 0
    df = df.replace('-', 0)
    
    # drop nan values
    df = df.dropna()
    
    # convert all columns to float
    for col in df.columns:
        if col != 'Team':
            df[col] = df[col].astype(float)
            
    # replace any inf with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # normalize the data
    for col in df.columns:
        if col != 'Team':
            df[col] = (df[col] - df[col].mean()) / df[col].std() # column = (column - mean) / std
        
    return df

# normalize the past data
past_home_df = normalizeData(past_home_df.drop(columns = ['Team'])) # drop team name
past_away_df = normalizeData(past_away_df.drop(columns = ['Team'])) # drop team name

# normalize the current data
current_home_df = normalizeData(current_home_df)
current_away_df = normalizeData(current_away_df)

# =============================================== [--- Split Data  ---] ===============================================

# split data into training and testing
from sklearn.model_selection import train_test_split

# my target variable is the goals for (GF/60) or goals against (GA/60)
X_train_home_gf, X_test_home_gf, y_train_home_gf, y_test_home_gf = train_test_split(past_home_df.drop(columns = ['GF/60']), past_home_df['GF/60'], test_size = 0.2, random_state = 42)
X_train_away_gf, X_test_away_gf, y_train_away_gf, y_test_away_gf = train_test_split(past_away_df.drop(columns = ['GF/60']), past_away_df['GF/60'], test_size = 0.2, random_state = 42)

X_train_home_ga, X_test_home_ga, y_train_home_ga, y_test_home_ga = train_test_split(past_home_df.drop(columns = ['GA/60']), past_home_df['GA/60'], test_size = 0.2, random_state = 42)
X_train_away_ga, X_test_away_ga, y_train_away_ga, y_test_away_ga = train_test_split(past_away_df.drop(columns = ['GA/60']), past_away_df['GA/60'], test_size = 0.2, random_state = 42)


# =============================================== [--- Ridge Regression  ---] ===============================================
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ---- Build Models -----

ridge_home_gf = Ridge(alpha = 1.0) # create the model (goals for)
ridge_home_ga = Ridge(alpha = 1.0) # create the model (goals against)

ridge_away_gf = Ridge(alpha = 1.0) # create the model (goals for)
ridge_away_ga = Ridge(alpha = 1.0) # create the model (goals against)

# ---- Fit Models -----

ridge_home_gf.fit(X_train_home_gf, y_train_home_gf) # fit the model
ridge_home_ga.fit(X_train_home_ga, y_train_home_ga) # fit the model

ridge_away_gf.fit(X_train_away_gf, y_train_away_gf) # fit the model
ridge_away_ga.fit(X_train_away_ga, y_train_away_ga) # fit the model

# ---- Predict -----

y_pred_home_gf = ridge_home_gf.predict(X_test_home_gf) # predict the goals for
y_pred_home_ga = ridge_home_ga.predict(X_test_home_ga) # predict the goals against

y_pred_away_gf = ridge_away_gf.predict(X_test_away_gf) # predict the goals for
y_pred_away_ga = ridge_away_ga.predict(X_test_away_ga) # predict the goals against

mse_home_gf = mean_squared_error(y_test_home_gf, y_pred_home_gf)
mse_home_ga = mean_squared_error(y_test_home_ga, y_pred_home_ga)
mse_away_gf = mean_squared_error(y_test_away_gf, y_pred_away_gf)
mse_away_ga = mean_squared_error(y_test_away_ga, y_pred_away_ga)

print("Home Mean Squared Error Goals For: " + str(mse_home_gf))
print("Home Mean Squared Error Goals Against: " + str(mse_home_ga))
print("Away Mean Squared Error Goals For: " + str(mse_away_gf))
print("Away Mean Squared Error Goals Against: " + str(mse_away_ga))

# =============================================== [--- Lasso Regression  ---] ===============================================
# from sklearn.linear_model import Lasso

# lasso = Lasso(alpha = 1.0) # create the model

# lasso.fit(X_train_home_gf, y_train_home_gf) # fit the model

# y_pred_home = lasso.predict(X_test_home_gf) # predict the goals for

# mse = mean_squared_error(y_test_home_gf, y_pred_home) # calculate the mean squared error
# print("Mean Squared Error: " + str(mse))


# =============================================== [--- Predict on Current Data  ---] ===============================================
# create a dataframe to hold the predictions
predictions_df = pd.DataFrame(columns = ['Team', 'Home Goals For', 'Away Goals For', 'Home Goals Against', 'Away Goals Against'])

predictions_df['Team'] = current_home_df['Team'] # add the team names
predictions_df = predictions_df.set_index('Team')

current_home_gf = current_home_df.drop(columns = ['GF/60', 'Team'])
current_away_gf = current_away_df.drop(columns = ['GF/60', 'Team'])
current_home_ga = current_home_df.drop(columns = ['GA/60', 'Team'])
current_away_ga = current_away_df.drop(columns = ['GA/60', 'Team'])

# use the ridge models to predict the goals for and goals against
current_home_gf_pred = ridge_home_gf.predict(current_home_gf)
current_away_gf_pred = ridge_away_gf.predict(current_away_gf)
current_home_ga_pred = ridge_home_ga.predict(current_home_ga)
current_away_ga_pred = ridge_away_ga.predict(current_away_ga)

# right now the predictions can be negative, so we need to make them all positive by adding the absolute value of the minimum prediction
min_pred = min(min(current_home_gf_pred), min(current_away_gf_pred), min(current_home_ga_pred), min(current_away_ga_pred))
current_home_gf_pred = current_home_gf_pred + abs(min_pred)
current_away_gf_pred = current_away_gf_pred + abs(min_pred)
current_home_ga_pred = current_home_ga_pred + abs(min_pred)
current_away_ga_pred = current_away_ga_pred + abs(min_pred)    

# add the predictions to the dataframe
predictions_df['Home Goals For'] = current_home_gf_pred
predictions_df['Away Goals For'] = current_away_gf_pred
predictions_df['Home Goals Against'] = current_home_ga_pred
predictions_df['Away Goals Against'] = current_away_ga_pred

# add an average row at the bottom for each column
predictions_df.loc['Average'] = predictions_df.mean()

# create a attack strength and defense strength column (for home and away) (value between 0 and 1)
def calculate_attack_strength(goals_for, average_goals_for, elo):
    return ((goals_for / average_goals_for) * elo) + 0.5

def calculate_defense_strength(goals_against, average_goals_against, elo):
    return ((goals_against / average_goals_against) / (elo * 4)) + 0.5

strengths_df = pd.DataFrame(columns = ['Home Attack Strength', 'Home Defense Strength', 'Away Attack Strength', 'Away Defense Strength'])
strengths_df['Team'] = predictions_df.index # add the team names
strengths_df = strengths_df.set_index('Team') # set the index to the team names

elo = get_elo()

# add elo to the strengths dataframe
strengths_df = add_elo(strengths_df, elo)

strengths_df['Home Attack Strength'] = predictions_df.apply(lambda row: calculate_attack_strength(row['Home Goals For'], predictions_df.loc['Average']['Home Goals For'], strengths_df.loc[row.name]['ELO']), axis = 1)
strengths_df['Home Defense Strength'] = predictions_df.apply(lambda row: calculate_defense_strength(row['Home Goals Against'], predictions_df.loc['Average']['Home Goals Against'], strengths_df.loc[row.name]['ELO']), axis = 1)
strengths_df['Away Attack Strength'] = predictions_df.apply(lambda row: calculate_attack_strength(row['Away Goals For'], predictions_df.loc['Average']['Away Goals For'], strengths_df.loc[row.name]['ELO']), axis = 1)
strengths_df['Away Defense Strength'] = predictions_df.apply(lambda row: calculate_defense_strength(row['Away Goals Against'], predictions_df.loc['Average']['Away Goals Against'], strengths_df.loc[row.name]['ELO']), axis = 1)

# right now the spread between the best and worst teams in each category is too large, so we need to scale it down
strengths_df['Home Attack Strength'] = strengths_df.apply(lambda row: row['Home Attack Strength'] / max(strengths_df['Home Attack Strength']), axis = 1)
strengths_df['Home Defense Strength'] = strengths_df.apply(lambda row: row['Home Defense Strength'] / max(strengths_df['Home Defense Strength']), axis = 1)
strengths_df['Away Attack Strength'] = strengths_df.apply(lambda row: row['Away Attack Strength'] / max(strengths_df['Away Attack Strength']), axis = 1)
strengths_df['Away Defense Strength'] = strengths_df.apply(lambda row: row['Away Defense Strength'] / max(strengths_df['Away Defense Strength']), axis = 1)

# get an overall strength for each team (high attack and low defense is good, low attack and high defense is bad) (value between 0 and 1)
strengths_df['Overall Strength'] = strengths_df.apply(lambda row: (row['Home Attack Strength'] + row['Away Attack Strength']) - (row['Home Defense Strength'] + row['Away Defense Strength']), axis = 1)

# shift the overall strength so its between 0 and 1
strengths_df['Overall Strength'] = strengths_df.apply(lambda row: row['Overall Strength'] + abs(min(strengths_df['Overall Strength'])), axis = 1)
strengths_df['Overall Strength'] = strengths_df.apply(lambda row: row['Overall Strength'] / max(strengths_df['Overall Strength']), axis = 1)


# =============================================== [--- Predict Games  ---] ===============================================
from scipy.stats import poisson

# using the strengths of each team, predict the outcome of each game
def predict_game(home_team, away_team):
    home_attack_strength = strengths_df.loc[home_team]['Home Attack Strength']
    home_defense_strength = strengths_df.loc[home_team]['Home Defense Strength']
    
    away_attack_strength = strengths_df.loc[away_team]['Away Attack Strength']
    away_defense_strength = strengths_df.loc[away_team]['Away Defense Strength']
    
    # calculate the expected goals for and goals against
    home_expected_gf = home_attack_strength * away_defense_strength * predictions_df.loc['Average']['Home Goals For']
    
    away_expected_gf = away_attack_strength * home_defense_strength * predictions_df.loc['Average']['Away Goals For']
    
    away_prob = 0
    home_prob = 0
    tie_prob = 0
    
    for i in range(0, 15):
        for j in range(0, 15):
            prob = poisson.pmf(i, home_expected_gf) * poisson.pmf(j, away_expected_gf)
            if i > j:
                home_prob += prob
            elif j > i:
                away_prob += prob
            else:
                tie_prob += prob
                
    away_prob = away_prob + (tie_prob / 2)
    home_prob = home_prob + (tie_prob / 2)
                
    return home_prob, away_prob

# print a heatmap of the team's strengths
def illustrate_strengths():
    plt.figure(figsize = (13, 8))
    sns.heatmap(strengths_df.sort_values(by = 'ELO', ascending = False), annot = True)
    plt.show()

# print(predict_game('Boston Bruins', 'Edmonton Oilers'))
# print(predict_game('Detroit Red Wings', 'Philadelphia Flyers'))
# print(predict_game('Boston Bruins', 'Colorado Avalanche'))

# illustrate_strengths()

# =============================================== [--- Odds ---] ===============================================

api_key = '74a13ca8f52c11c2476a5cc7db5d34d0' # api key for sports betting odds

def decimal_to_american(odds):
    if odds > 2:
        return int((odds - 1) * 100)
    else:
        return int(-100 / (odds - 1))

# returns the odds for the away team and the home team (decimal)
def get_odds(away_team, home_team):
    away_prob, home_prob = predict_game(away_team, home_team)
    
    away_odds = round(1 / away_prob, 2)
    home_odds = round(1 / home_prob, 2)
    
    return away_odds, home_odds

def clean_odds():
    odds_response = requests.get(f'https://api.the-odds-api.com/v3/odds/?sport=icehockey_nhl&region=us&mkt=h2h&dateFormat=iso&apiKey={api_key}')
    
    odds_json = json.loads(odds_response.text)['data']
    
    simple_odds = []

    for game in odds_json:
        home_team = game['home_team']
        # away team is the team in teams that is not the home team
        away_team = [team for team in game['teams'] if team != home_team][0]
        commence_time = game['commence_time']
        
        away_odds = game['sites'][0]['odds']['h2h'][0]
        home_odds = game['sites'][0]['odds']['h2h'][1]
        
        simple_odds.append([home_team, away_team, home_odds, away_odds, commence_time])
        
    return simple_odds

def calculate_picks(odds):
    for game in odds:
        given_home_odds = game[2]
        given_away_odds = game[3]
        
        my_home_odds, my_away_odds = get_odds(game[1], game[0])
        
        # if the odds are better than the given odds, then bet on the team
        if my_away_odds < given_away_odds and (given_away_odds - my_away_odds) > 0.2:
            print(f"Bet on {game[1]} to win against {game[0]}")
            print(f"Given odds: {decimal_to_american(given_away_odds)}, My odds: {decimal_to_american(my_away_odds)}")
            print("-" * 50)
        elif my_home_odds < given_home_odds and (given_home_odds - my_home_odds) > 0.12:
            print(f"Bet on {game[0]} to win against {game[1]}")
            print(f"Given odds: {decimal_to_american(given_home_odds)}, My odds: {decimal_to_american(my_home_odds)}")
            print("-" * 50)
        
        
odds = clean_odds()
calculate_picks(odds)

# print(get_odds('Boston Bruins', 'Edmonton Oilers'))
# print(get_odds('Detroit Red Wings', 'Philadelphia Flyers'))
# print(get_odds('Boston Bruins', 'Colorado Avalanche'))


print("Code Completed.")
