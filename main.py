# Roblox EDA

import pandas as pd
import numpy as np

"""
Questions/Hypotheses to explore:

Games that include a reference to an update
in the start of their title will have a greater
ratio of current players to total players than
games that do not include a reference to an update
at the start of their title. This includes games
that reference the update at the end of their title.

There is a quadratic proportion between 
two variables in this dataset.
(Linear Regression / Poly Features)

Is the length of a title an indicator of the rating
of the game above 90%?
(Logistic Regression)

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
        """
        
        # Add ratio and clean to find updated games
        self.df['Retention'] = self.df['Active'] / self.df['Visits']
        # Regex is hard 
        self.update_df = self.df[self.df['Name'].str.match(r"^\[[^\]]*\][^\[]*$")]
        # Avoiding scientific notation and finding out information about the data
        pd.set_option('display.float_format', '{:.8f}'.format)
        whole_data = self.df['Retention'].agg([np.mean, min, max, np.median])
        update_data = self.update_df['Retention'].agg([np.mean, min, max, np.median])
        print(whole_data)
        print(update_data)
        print(self.df[self.df['Retention'] > 0.007])
        
data = RobloxEDA()
data.hypothesis1()