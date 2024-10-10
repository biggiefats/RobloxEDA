# Roblox EDA

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

"""
Questions/Hypotheses LEFT to explore:

Are there common words used in games that have a high
all-time visits and a high favourites count?
(Clustering)

What games have disproportionate like and dislike counts,
given their ratio?
(DBSCAN)

Do the likes, dislikes and total visits indicate whether
or not a game's current players is above 10% of the number
of favourites on game?
(Naive Bayes)

Classify games by how many UNIQUE letters they have.
Does the general attributes of success to a game define
the unique letters in the game?
(Decision Tree Classifier)
"""

class RobloxEDA:
    def __init__(self):
        """
        Class used to handle the hypotheses testing and question answering.
        """
        
        self.df = pd.read_csv(r'roblox_games.csv', index_col=0)
        
        # Quantify data    
        for field in list(self.df.columns)[1:6]:
            self.df[field] = self.df[field].apply(lambda num: int("".join(num.split(","))))
            
    def hypothesis1(self):
        """
        Hypothesis:
        Games that include a reference to an update
        in the start of their title will have a greater
        ratio of current players to total players than
        games that do not include a reference to an update
        at the start of their title. This includes games
        that reference the update at the end of their title.
        
        Result:
        The hypothesis holds as being true - the mean of the
        overall data's ratio is 0.00014465 and the mean of
        the data for the updated games is 0.00020136. This,
        arguably could mean that the updated games have had
        recent players contribute better to their total visits.
        The median ratio for updated games is higher by a similar
        amount.
        
        Additional Findings:
        Both datasets have the same maximum ratio of
        0.00706515, which is from the game
        '[FREE TEST] Sorcery'. It has 26,186 active
        (according to the database) and has 3,706,360
        total visits, and is sat at #36 in terms of active
        players. It stands as one of the anomalies of their
        retention, for most games near that rank have above
        500,000,000 visits. 
        
        Upon research, this is due to the
        love that Roblox players have for games that are inspired
        on Anime (japanese animation) - this game capitalizes of 
        such medium of entertainment.
        """
        
        # Add ratio and clean to find updated games
        self.df['Retention'] = self.df['Active'] / self.df['Visits']
        # Regex is hard 
        self.update_df = self.df[self.df['Name'].str.match(r"^\[[^\]]*\][^\[]*$")]
        # Avoiding scientific notation and finding out information about the data
        pd.set_option('display.float_format', '{:.8f}'.format)
        whole_data = self.df['Retention'].agg([np.mean, min, max, np.median])
        update_data = self.update_df['Retention'].agg([np.mean, min, max, np.median])
        
        # Showing data
        # print(whole_data)
        # print(update_data)
        # print(self.df[self.df['Retention'] > 0.007])
        
    def hypothesis2(self):
        """
        Hypothesis:
        There is a quadratic proportion between 
        two variables in this dataset.
        
        Result:
        The two variables I decided to pick, based on visual understanding,
        was the relationship between likes and dislikes.
        Surprisingly, when we compare the quadratic regression line with
        a linear regression line, they achieve a similar 'r2_score' (e.g.
        how well they fit with the line); their difference in success is
        0.000099030, to 9dp (essentially 0.0001, which is 0.01%), so these
        variables actually fit a linear regression line. I believe that this
        is due to how dislikes becomes more spread out with more likes and then
        there are the odd outperforming games that have extremes of likes and
        dislikes, so the regression algorithm is biased towards the cluster
        of points with low likes and dislikes.
        
        Additional Findings:
        The game with the 4th highest dislikes is Tower of Hell, a game
        with focuses on players having to reach the top of a tower by completing
        an obstacle course (think Total Wipeout, Ninja Warrior, Takeshi's Castle).
        The games in the top 5, with this game, have ranks (11, 2, 6, 47) that are relatively
        close to the top games in dislikes but this game is a bit further out. This is due
        to the rage that is induced by such a game; when you make a mistake, you fall down
        the tower, which is heavily taxing on patience. Games such as Getting Over It and
        Only Up! share this gameplay loop.
        
        The game, Brookhaven RP, is in the top 5 for both likes and dislikes. This game is
        a role-playing game where you essentially simulate a life as a human, similar to The
        Sims, but with more interaction from the player. The game might have a lot of likes
        and dislikes due to the same reasons: roleplaying, player relationships, creative
        requirements (the experience is driven by the player heavily, rather than the game
        providing the creativity) or other aspects beyond the surface. It may be hard for
        me to pinpoint the real reasons, as those games are harder to decipher and understand
        from a layman's point of view.
        """
        # seaborn.pairplot will pair all possible variables
        # sns.pairplot(data=self.df, height=1.2)
        
        # check aggregate functions to scale data, so axis values are human readable
        # print(self.df['Likes'].mean()) - 316879.215
        # print(self.df['Dislikes'].mean()) - 4230.899
        x = self.df['Likes'] // 4000
        X = x.values.reshape(-1,1)
        y = self.df['Dislikes'] // 4000
        
        # highest dislikes and likes
        top_x = self.df.loc[x.sort_values(ascending=False).head(5).index]
        top_y = self.df.loc[y.sort_values(ascending=False).head(5).index]
        # print(top_x)
        # print()
        # print(top_y)
        
        # algorithms
        # quadratic regression line
        quadratic = PolynomialFeatures(degree=2, include_bias=False)
        quadratic_X = quadratic.fit_transform(X)
        model = LinearRegression()
        model.fit(quadratic_X, y)
        quad_y = model.predict(quadratic_X)
        
        # linear regression line
        model = LinearRegression()
        model.fit(X, y)
        linear_y = model.predict(X)
        
        difference_in_models = abs(r2_score(y, quad_y) - r2_score(y, linear_y))
        # print(f"{difference_in_models:.9f}") - 0.0001
        
        # data visualization
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_title("Likes against Dislikes for Roblox Games")
        ax.set_xlabel("Likes (4000s)")
        ax.set_ylabel("Dislikes (4000s)")
        ax.plot(X, quad_y, color='cyan') # quadratic regression line
        ax.plot(X, linear_y, color='red') # linear regression line
        sns.scatterplot(x=x, y=y, hue=self.df['Likes'], palette='husl', ax=ax)
        plt.show()
        
    def hypothesis3(self):
        """
        Question:
        Is the length of the title an indicator of the rating being above 90%?
        
        Result:
        The model's score is 67%. This is an indication that the length
        of a title does not indicate a game as having a rating greater than 90%.
        An ideal score would be above 85%, which this is not.
        
        The logistic regression line, as the length of titles increases, assumes that
        there is an increasing chance that the game has a rating greater than 90%.
        
        Additional Findings:
        The (presumed) ideal length for a title to get a 90.0+ rating is to have the title
        between 16 and 19 characters long. Surprisingly, 19 games that have a 90.0+ rating
        have 19 characters in their title. The mean length of a title is also the median
        length of a title but is NOT the mode length of a title, which doesn't make it
        qualifiable as a normal distribution.
        """
        # Defining variables
        self.df['High Rating'] = (self.df['Rating'] > 90).astype(int)
        self.df['Length of Title'] = list(map(lambda x: len(x), self.df['Name']))
        
        x = self.df['Length of Title']
        X = x.values.reshape(-1,1)
        y = self.df['High Rating']
        
        # Logistic regression
        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(X, y)
        a = model.intercept_[0]
        b = model.coef_[0][0]
        # Equation of logistic curve
        logistic_y = 1 / (1 + np.exp(-(a + b * self.df['Length of Title'])))
        
        # Stats
        # Making series to show the count of games that were 90.0+ in rating, for each possible length
        # print(model.score(X, y))
        rating_length_dict = self.df[self.df['High Rating'] == 1].groupby(by='Length of Title').groups
        item_frequency = [len(value) for value in rating_length_dict.values()]
        freq_df = pd.Series(item_frequency, index=rating_length_dict.keys())
        print(pd.Series(rating_length_dict.keys()).agg([min, max, np.median, np.std, np.mean]))
        # print(freq_df)
        
        # Visualisation
        fig = plt.figure(figsize=(9,6))
        fig.suptitle("Does the Length of a Title Correlate with High Rating?")
        # Logistic Regression 'Curve' with scatter plot
        ax = fig.add_subplot(121)
        ax.set_xlabel("Length of Title (characters)")
        ax.set_ylabel("High Rating? (Above 90.0)")
        sns.scatterplot(data=self.df, x='Length of Title', y='High Rating', hue='Rating', palette='husl', ax=ax)
        ax.plot(self.df['Length of Title'], logistic_y, color='black')
        
        # Frequency plot of length of titles
        ax = fig.add_subplot(122)
        ax.set_xlabel("Length of Title (characters)")
        ax.set_ylabel("Frequency")
        ax.plot(freq_df.index, freq_df.values, color='black')
        plt.show()
        
data = RobloxEDA()
data.hypothesis3()