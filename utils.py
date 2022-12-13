#general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
#ML imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# All my cleaning funtions

def cleanGen(games_df, ranked_df):
    #turns mundo into Dr.
    for value in ranked_df["Champion"]:
        if "mundo" in value.lower():
            ranked_df.replace(value, "Dr. Mundo", inplace=True)
    #fixes result name
    for value in ranked_df["Result"]:
        if "loss" in value.lower():
            ranked_df.replace(value, "Defeat", inplace=True)
    #removes N/A
    games_df.replace("N/A", np.NaN, inplace=True)
    #makes an aram df in case i want to use it later
    aram_df = games_df.loc[games_df["Gamemode"] == "ARAM"]
    #although only 2 remakes, it could mess up the data
    remove_remakes = games_df.loc[games_df["Result"] == "Remake"]
    #removes both from games_df
    games_df = games_df.drop(aram_df.index)
    games_df = games_df.drop(remove_remakes.index)

    return games_df, ranked_df, aram_df


def cleanStats(stats_df):
    #Fixes the names
    for name in stats_df["Champion"]:
        length = len(name)
        split_index = length // 2
        #sets val equal to first half
        stats_df.replace(name, name[:split_index], inplace=True)
    
    #removes the "%" from the values and turns them into ints
    for stat in stats_df["Role %"]:
        #sets val equal to first half
        stats_df.replace(stat, float(stat[:5]), inplace=True)
    
    #removes all the champs with a roll % less than 50
    to_drop = stats_df.loc[stats_df["Role %"] < 50]
    stats_df = stats_df.drop(to_drop.index)
    
    #drops unnecessary columns
    colums_td = ["Score", "Trend", "Pick %", "Ban %"]
    stats_df = stats_df.drop(columns=colums_td)
    #for some reason these wouldnt drop when I put them in columns_td
    stats_df = stats_df.drop(columns="Role")
    stats_df = stats_df.drop(columns="Role %")
    
    stats_df.rename(columns={'KDA': 'Avg KDA of Champ'}, inplace=True)
    
    return stats_df



# All my graphing functions

def graphRanked(ranked_df):
    grouped_by_champ = ranked_df.groupby("Champion")

    print(grouped_by_champ.size())

    plt.figure(figsize=(8,8), facecolor="white")

    xs = ["Dr. Mundo", "Lillia", "Mordekaiser", "Vi"]

    plt.title("Champions Played in Ranked")
    plt.pie(grouped_by_champ.size(), labels=xs, autopct="%1.1f%%")
    plt.show()


    mundo_df = grouped_by_champ.get_group("Dr. Mundo")
    lil_df = grouped_by_champ.get_group("Lillia")
    mord_df = grouped_by_champ.get_group("Mordekaiser")
    vi_df = grouped_by_champ.get_group("Vi")

    # reset figure
    plt.figure()

    ys = [mundo_df["KDA"].mean(), lil_df["KDA"].mean(), mord_df["KDA"].mean(), vi_df["KDA"].mean()]

    xrng = np.arange(len(xs))
    yrng = np.arange(0, max(ys)+1, 1)

    plt.bar(xrng, ys, 0.45, align="center") 
    # note: default alignment is center

    plt.xlabel("Champions")
    plt.ylabel("Average KDA")
    plt.title("Avg KDA per Champion in Ranked")

    plt.xticks(xrng, xs)
    plt.yticks(yrng)
    # turn on the background grid
    plt.grid(True)
    plt.show()


def graphNorms(games_df):

    #Pi Chart
    grouped_by_roll = games_df.groupby("Roll")

    print(grouped_by_roll.size())

    plt.figure(figsize=(8,8), facecolor="white")

    xs = ["ADC", "Jungle", "Middle", "Support", "Top"]

    plt.title("Rolls Played in Norms")
    plt.pie(grouped_by_roll.size(), labels=xs, autopct="%1.1f%%")
    plt.show()


    #Bar Chart
    adc_df = grouped_by_roll.get_group("ADC")
    jng_df = grouped_by_roll.get_group("Jungle")
    mid_df = grouped_by_roll.get_group("Middle")
    sup_df = grouped_by_roll.get_group("Support")
    top_df = grouped_by_roll.get_group("Top")

    # reset figure
    plt.figure()

    ys = [adc_df["KDA"].mean(), jng_df["KDA"].mean(), mid_df["KDA"].mean(), sup_df["KDA"].mean(), top_df["KDA"].mean()]


    xrng = np.arange(len(xs))
    yrng = np.arange(0, max(ys)+1, 1)

    plt.bar(xrng, ys, 0.45, align="center") 
    # note: default alignment is center

    plt.xlabel("Rolls")
    plt.ylabel("Average KDA")
    plt.title("Avg KDA per Roll in Norms")

    plt.xticks(xrng, xs)
    plt.yticks(yrng)
    # turn on the background grid
    plt.grid(True)
    plt.show()


def graphTime(games_df, ranked_df):

    plt.figure()

    xs = ["Norms", "Ranked"]
    plt.boxplot([games_df["Length (min)"], ranked_df["Length (min)"]])


    plt.ylabel("Length (min)")
    plt.title("Length of Norms vs. Ranked games")

    plt.show()



# Hypothysis Testing

def akaliHypo(games_df):

    grouped_by_champ = games_df.groupby("Champion")
    akali_df = grouped_by_champ.get_group("Akali")

    akali_kda = akali_df["KDA"]

    t = (akali_kda.mean() - 2.31)/(akali_kda.std()/math.sqrt(len(akali_kda)))
    print("t-computed =", t)

    print("\nAverage KDA of Akalis: ", akali_df["Avg KDA of Champ"].mean())


def friendsHypo(games_df):

    #this makes a new df to clean that turns the ranks into only 2 different values
    avgranks_df = games_df.copy()
    for rank in avgranks_df["Average Rank"]:
        if "Gold" in rank:
            avgranks_df.replace(rank, "Higher", inplace=True)
        if "Plat" in rank:
            avgranks_df.replace(rank, "Higher", inplace=True)
        if "Bronze" in rank:
            avgranks_df.replace(rank, "MyElo", inplace=True)
        if "Silver" in rank:
            avgranks_df.replace(rank, "MyElo", inplace=True)

    #this will now split the df into 2, one with a higher avg rank, and one with about the same
    grouped_by_rank = avgranks_df.groupby("Average Rank")
    friends_df = grouped_by_rank.get_group("Higher")
    alone_df = grouped_by_rank.get_group("MyElo")

    t, p = stats.ttest_ind(friends_df["Length (min)"], alone_df["Length (min)"])
    print("t-calculated =", t)



# Machine Learning

def labelEncoding(games_df):

    to_drop = ["Gamemode", "Tier", "Win %", "Avg KDA of Champ"]
    games_ml = games_df.drop(columns=to_drop)


    #kindof a roundabout way of encoding both of these, but it works

    #Also going to encode the new games df here as well
    newgames_df = pd.read_csv("new_games.csv", header = 0, index_col="ID")
    newgames_df = newgames_df.drop(columns="Gamemode")

    #adding the two df's together to get the same encoding
    games_ml = games_ml.append(newgames_df)

    #label encoding
    le = LabelEncoder()
    to_encode = ["Champion", "Roll", "Average Rank"]

    le.fit(games_ml["Result"])
    games_ml["Result"] = le.fit_transform(games_ml["Result"])

    games_ml[to_encode] = games_ml[to_encode].apply(le.fit_transform)

    #seperates the dfs
    newgames_df = games_ml[88:98]
    games_ml = games_ml[0:87]

    return games_ml, newgames_df


def decisionTree(X, y):

    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    # tree?
    fig = plt.figure(figsize=(15,10))
    plot_tree(clf, feature_names=X.columns, class_names={1: "Victory", 0: "Defeat"}, filled=True)

    plt.show()

    #returns clf for later use
    return clf


def accuracyTesting(X, y, clf):

    # scaling X
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    #did some testing and 11 was the best for kNN
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)

    # accuracy test
    r2score = knn.score(X_test, y_test)
    print("kNN score:", r2score)


    #Decision Tree score from early decision tree
    r2score = clf.score(X_test, y_test)
    print("Decision Tree score:", r2score)

    #returns knn for later use
    return knn


def mlPredictions(newgames_df, knn, clf):

    results = newgames_df["Result"] 
    newgames_df = newgames_df.drop(columns="Result")

    knnpredictions = knn.predict(newgames_df)
    print("kNN Predictions", knnpredictions)
    dtpredictions = clf.predict(newgames_df)
    print("Decision Tree Predictions", dtpredictions)
    print("\nReal Results")
    for i in results:
        print(i, end=" ")


    #Finds the accuracy of our predicitons
    percent = 0
    for i in range(10):
        if(knnpredictions[i] == results[i+1]):
            percent = percent + 1
    print("\n\nkNN Accuracy -", percent*10, "%")
    percent = 0
    for i in range(10):
        if(dtpredictions[i] == results[i+1]):
            percent = percent + 1
    print("Decision Accuracy -", percent*10, "%")