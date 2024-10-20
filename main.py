# Roblox EDA

import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

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
        
        self.df['Retention'] = self.df['Active'] / self.df['Visits']
        # Regex brainrot 
        self.update_df = self.df[self.df['Name'].str.match(r"^\[[^\]]*\][^\[]*$")]
        pd.set_option('display.float_format', '{:.8f}'.format)
        whole_data = self.df['Retention'].agg([np.mean, min, max, np.median])
        update_data = self.update_df['Retention'].agg([np.mean, min, max, np.median])
        
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
        # sns.pairplot(data=self.df, height=1.2)
        
        # print(self.df['Likes'].mean()) - 316879.215
        # print(self.df['Dislikes'].mean()) - 4230.899
        x = self.df['Likes'] // 4000
        X = x.values.reshape(-1,1)
        y = self.df['Dislikes'] // 4000
        
        top_x = self.df.loc[x.sort_values(ascending=False).head(5).index]
        top_y = self.df.loc[y.sort_values(ascending=False).head(5).index]
        # print(top_x)
        # print()
        # print(top_y)
        
        quadratic = PolynomialFeatures(degree=2, include_bias=False)
        quadratic_X = quadratic.fit_transform(X)
        model = LinearRegression()
        model.fit(quadratic_X, y)
        quad_y = model.predict(quadratic_X)
        
        model = LinearRegression()
        model.fit(X, y)
        linear_y = model.predict(X)
        
        difference_in_models = abs(r2_score(y, quad_y) - r2_score(y, linear_y))
        # print(f"{difference_in_models:.9f}") - 0.0001
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_title("Likes against Dislikes for Roblox Games")
        ax.set_xlabel("Likes (4000s)")
        ax.set_ylabel("Dislikes (4000s)")
        ax.plot(X, quad_y, color='cyan')
        ax.plot(X, linear_y, color='red')
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
        self.df['High Rating'] = (self.df['Rating'] > 90).astype(int)
        self.df['Length of Title'] = list(map(lambda x: len(x), self.df['Name']))
        
        x = self.df['Length of Title']
        X = x.values.reshape(-1,1)
        y = self.df['High Rating']
        
        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(X, y)
        a = model.intercept_[0]
        b = model.coef_[0][0]
        logistic_y = 1 / (1 + np.exp(-(a + b * self.df['Length of Title'])))
        
        # Making series to show the count of games that were 90.0+ in rating, for each possible length for a title
        # print(model.score(X, y))
        rating_length_dict = self.df[self.df['High Rating'] == 1].groupby(by='Length of Title').groups
        item_frequency = [len(value) for value in rating_length_dict.values()]
        freq_df = pd.Series(item_frequency, index=rating_length_dict.keys())
        print(pd.Series(rating_length_dict.keys()).agg([min, max, np.median, np.std, np.mean]))
        # print(freq_df)
        
        fig = plt.figure(figsize=(9,6))
        fig.suptitle("Does the Length of a Title Correlate with High Rating?")
        ax = fig.add_subplot(121)
        ax.set_xlabel("Length of Title (characters)")
        ax.set_ylabel("High Rating? (Above 90.0)")
        sns.scatterplot(data=self.df, x='Length of Title', y='High Rating', hue='Rating', palette='husl', ax=ax)
        ax.plot(self.df['Length of Title'], logistic_y, color='black')
        
        ax = fig.add_subplot(122)
        ax.set_xlabel("Length of Title (characters)")
        ax.set_ylabel("Frequency")
        ax.plot(freq_df.index, freq_df.values, color='black')
        plt.show()
    
    def hypothesis4(self):
        """
        Question:
        Are there common words used in games that have a high
        all-time visits and a high favourites count?
        
        Results:
        Selecting the top 150 games, we find that:
        
        1ST: 'update' and 'simulator'
        'update' being 1st increases the validity of the first hypothesis
        method; games mentioning the word 'update' do have better
        metrics. This is to be expected with all-time visits being
        a metric of success for this question, as updates promotes
        loyalty to a game.
        
        'simulator' is a genre of game that has a player take the role
        of someone in a simulated environment such that there is a long
        progression system involved in the game with elements including
        luck, achievements, skill trees or other 'grinding' elements.
        These games promote multiple visits as certain parts of the game
        have players play the game for multiple hours in a singular session.
        Simulator games also have players favourite the game for certain
        items and perks, which are useful EARLY in the game, where a player
        is most desperate to get something.
        
        2ND: 'the'
        As this is a determiner, it would be expected to find this word 
        in the top words. Words such as 'of', 'a', 'to' and 'and' are 
        also of similar nature; they allow for a more rigid game title.
        
        3RD: 'rp' and 'tycoon'
        'rp' is an abbreviation of roleplay. roleplaying games are the
        same as how I described simulator games expect that there isn't
        as much progression required - the progression is more how you define
        it whereas in a simulator, certain metrics define progression. For
        kids, it may be expected that they like to ideate and imagine things
        due to their stimulation to things and their minds not knowing
        limits well.
        
        'tycoon' games are bound to an area usually being the part of progression
        (e.g. upgrade a house by adding things in the house) and have players
        passively generate a currency to progress. The 'grinding' aspect promotes
        people to play the game multiple times.
        
        Additional Findings:
        Other words from the top 25 words include 'tower', 'x',
        'car', 'legends', 'life', 'limited', 'obby', 'speed' and 'survive'.
        The 'x' alludes to a multiplier to a statistic in a game on top of
        its base increment, or it can refer to a collaboration of different
        developers, groups or companies.
        
        Furthermore, when we look at ALL entires, the trends change slightly:
        Here are the top 10 words:
        1) 'simulator'
        2) 'the'
        3) 'obby'
        4) 'tycoon'
        5) 'update'
        6) 'new'
        7) 'rp'
        8) 'a'
        9) 'upd'
        10) 'x'
        
        A word that wasn't originally in the top games was 'obby'. This may
        be due to the fall of obby games in modern Roblox, which were popular
        in the earlier stages of Roblox.
        """   
        
        secret_sums = list()
        favourites_df = self.df.sort_values(by=['Favourites'], ascending=False).reset_index()
        visits_df = self.df.sort_values(by=['Visits'], ascending=False).reset_index()
        
        # Get the ranking for each game by making a secret sum that looks at favourites and visits
        for title in self.df['Name']:
            favourite_score = list(favourites_df[favourites_df['Name'] == title]['Rank'].replace('#', '').index.astype(int))[0]
            visit_score = list(visits_df[visits_df['Name'] == title]['Rank'].replace('#', '').index.astype(int))[0]
            secret_sum = favourite_score + visit_score
            secret_sums.append(secret_sum)
        
        self.df['secret_sum'] = pd.Series(secret_sums).values
        top_games_df = self.df.sort_values(by='secret_sum').reset_index().iloc[:, :8]
        
        word_frequency_map = defaultdict(int)
        word_frequency_df = pd.DataFrame(columns=['Word', 'Count'])
        
        for title in top_games_df['Name']:
            title_words = title.split(' ')
            # Get digits and emojis out and add it to word_frequency_map
            for word in title_words:
                word = re.sub(r'[^\w]*|[\d]*', '', word).lower()
                if word.isalpha():
                    word_frequency_map[word] += 1
                    
        word_frequency_map = dict(sorted(word_frequency_map.items(), key=lambda item: item[1], reverse=True))
        for key, value in word_frequency_map.items():
            word_frequency_df.loc[len(word_frequency_df)] = key, value
        
        fig = plt.figure(figsize=(12,9))
        fig.suptitle("Words Used in Games with High All-Time Visits and Favourites Count")
        ax = fig.add_subplot(111)
        ax.set_title("Most Frequent Words")
        sns.barplot(data=word_frequency_df.head(25), x='Word', y='Count', hue='Count', palette='viridis', ax=ax)
        ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=6)
        plt.show()
        
    def hypothesis5(self):
        """
        Question:
        What games have disproportionate like and dislike counts?
        
        Answer:
        I decided to use DBSCAN to detect outliers.
        
        DBSCAN detects 5 clusters:\n
        -1 (Noise): 23\n
        0 (Core): 961\n
        1 (Outlying Group 1): 5\n
        2 (Outlying Group 2): 5\n
        3 (Outlying Group 3): 6\n
        
        The reason by Cluster -1 is called 'noise' is that DBSCAN is an acronym for
        'Density-Based Spatial Clustering of Applications with Noise'. The noise refers
        to any point that could not be reached via a proximity radius generated
        from any core point.
        
        All the games that are considered to be noise, that are in Cluster -1, are 
        games that are beloved to the platform, and have existed for some amount of 
        time. The outlier in the noise cluster would be 'Speed Run 4 🍩😋[UPDATE]', 
        as it sits near the games of cluster 1, due to its low like and dislike count.
        
        Clusters 1, 2 and 3 all sit near each other in terms of their like counts,
        but are further enough from the games in cluster 0 to be considered outliers.
        
        Cluster 1 has games that have around 2.25 million likes, which are RIVALS,
        Bee Swarm Simulator, Flee The Facility, PLS DONATE and Phantom Forces.
        These games all are doing quite well in the ranks, sitting beyond the top 200
        games on Roblox in activity. They all also have a like to dislike ratio of
        92+ and all have over 1 billion visits expect RIVALS, which is a fairly new
        game with 837 million visits.
        
        Cluster 2 has games that have around 1.95 million likes, which are Anime Defenders,
        Evade, Shindo Life, Death Ball and Anime Dimensions Simulator. All these games are
        the top games of the platform (above top 100) expect Anime Dimensions Simulator,
        which may be an old game due to it sharing similar statistics with the rest
        of these games.
        
        Cluster 3 has games with games near 1.9 million likes, with little variance.
        The games include Theme Park Tycoon 2, Funky Friday, Lumber Tycoon 2,
        Epic Minigames, Mad City: Chapter 2 and Brreaking Point. None of these games
        are in the top 100, indicating that these games used to be popular but are now
        not, as they all have similar statistics to games in Cluster 2. The ratios
        sit between 87 and 90, with two games having 83 and 81 respectively.
        3 of the 6 games in Cluster 3 contain the number '2' in the title.
        """
        def elbow(X):
            # Find 4 nearest neighbours for all values in X
            nn = NearestNeighbors(n_neighbors=4)
            nn.fit(X)
            # Finds distances and positions for each neighbour
            distances, indices = nn.kneighbors()
            sorted_distances = np.sort(distances, axis=0)
            
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_xlabel("Point Number")
            ax.set_ylabel("Distance from Furthest Point")
            ax.plot(sorted_distances[:, 3])
            ax.axhline(y=1.34e+05, linestyle="dotted", color="black") # Measurement purposes
            plt.show()
        
        df = self.df
        # Find elbow / optimal Ɛ | optimal = approx. 134,000
        # elbow(df[['Dislikes', 'Likes']])
        
        model = DBSCAN(eps=1.34e+05, min_samples=4)
        model.fit(df[['Dislikes', 'Likes']])
        df['Cluster'] = model.labels_

        fig = plt.figure(figsize=(12,9))
        fig.suptitle("Likes and Dislikes for Each Game")
        ax = fig.add_subplot(111)
        sns.scatterplot(data=df, x='Likes', y='Dislikes', hue='Cluster', palette='viridis', ax=ax)
        plt.show()
        
        counts = df.groupby(by='Cluster').count()
        for group in sorted(list(df['Cluster'].unique())):
            if group != 0:
                print(f"\nGROUP {group}:")
                print(df.query(f"Cluster == {group}"))
        
    def hypothesis6(self):
        """
        Question:
        Do the likes, dislikes and total visits indicate whether
        or not a game's current players is above 10% of the number
        of favourites on game?

        Results:
        (forgive me for nesting functions into the method)
        
        Using Naive Bayes is fairly inaccurate, averaging less than
        50% success. This could mean that the answer to the question
        is no. However, Naive Bayes classification may not work well
        with such a dataset, for there are a lot more games that do
        not satisfy the condition of favourites. Naive Bayes uses
        probabilistic classification by weighing the chances of
        events happening relative to other events. Furthermore,
        Naive Bayes works better with independent outcomes. However,
        some games may have similar popularity because the games
        are extensions of others, or utilise a brand to make
        multiple games.
        
        Using PCA and Logistic Regression as well as a scaler
        works better than the Naive Bayes classification, averaging
        out a 93% successs rate, but the model predicts that ALL the 
        games do not satisfy the criteria. This may be a result of
        a majority of games not having active players be above 1/10
        of the total favourites. This is also a result of how linear
        regression models work, for they would be similar to a Bernoulli
        Naive Bayes classification model; two outcomes are only considered.
        
        A better method to use may be Decision Trees or Clustering, and that
        may inform whether or not the exogenous variables can predict
        the endogenous outcome.
        
        Therefore, based on the models used, the likes, dislikes and total
        visits do not indicate whether or not a game's current
        players is above 10% of the number of favourites on game.
        """
        
        def naive_bayes_method():
            """
            Using naive bayes to answer the question.
            """
            
            X = self.df[['Visits', 'Likes', 'Dislikes']]
            y = pd.Series(self.df['Active'].astype(int) > (self.df['Favourites'].astype(int) / 10)).values

            # Best to compare Multinomial, Complement and Bernoulli
            # We do not know about how normal this data is; rule out GaussianNB
            # We don't have categories of labels, only two possibilites; rule out CategoricalNB
            
            def best_model():
                # They all score the same, so pick any
                scores = np.zeros((3,1))
                for i in range(20):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)
                    
                    model1 = MultinomialNB()
                    model1.fit(X_train, y_train)
                    y1_pred = model1.predict(X_test)
                    y1_score = accuracy_score(y_test, y1_pred)
                    
                    model2 = ComplementNB()
                    model2.fit(X_train, y_train)
                    y2_pred = model1.predict(X_test)
                    y2_score = accuracy_score(y_test, y2_pred)
                    
                    model3 = BernoulliNB()
                    model3.fit(X_train, y_train)
                    y3_pred = model1.predict(X_test)
                    y3_score = accuracy_score(y_test, y3_pred)
                    
                    scores[0] += y1_score
                    scores[1] += y2_score
                    scores[2] += y3_score
                    
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)
            
            model = ComplementNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            C = confusion_matrix(y_true=y_test, y_pred=y_pred)
            display = ConfusionMatrixDisplay(confusion_matrix=C, display_labels=list(set(y_pred)))
            display.plot()
            plt.show()
        
        def pca_logistic_scaler_method():
            """
            Using PCA, logistic regression and a scaler to answer
            the question.
            """
            
            X = self.df[['Visits', 'Likes', 'Dislikes']]
            y = pd.Series(self.df['Active'].astype(int) > (self.df['Favourites'].astype(int) / 10)).values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            pca = PCA(0.975)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            
            model = LogisticRegression(solver='saga', random_state=0, tol=0.1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            C = confusion_matrix(y_true=y_test, y_pred=y_pred)
            display = ConfusionMatrixDisplay(confusion_matrix=C, display_labels=["False", "True"])
            display.plot()
            plt.show()
            
            print(accuracy_score(y_true=y_test, y_pred=y_pred))
                   
    def hypothesis7(self):
        """
        Question:
        Classify games by how many UNIQUE letters they have.
        
        Does the general attributes of success to a game define
        the unique letters in the game?
        
        Results:
        Upon realisation, a Decision Tree is not a good model
        for this question. The endogenous variable has too many
        categories, which would make the tree very large. It has
        an average success rate of 11%.
        
        Using a logistic regression method also returns a similar
        success rate.
        
        Therefore, it would be assumed that the general attributes
        of a game does not determine the unique letters of game.
        
        Additional Findings:
        When plotting the lengths in order, with an arbitrary
        x axis, we get a sigmoid curve.
        """
        X = self.df.iloc[:, 1:]
        y = self.df['Name'].str.lower().str.replace('[^a-z]', '', regex=True).apply(lambda x: len(set(x))).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)
        
        def decision_tree_method_7():
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            plot_tree(decision_tree=model, filled=True, rounded=True)
    
            # plt.show() # Remove if your PC is not good
            
        def visualise_7():
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_title("Logistic Curve via Sorted Lengths and Counts")
            ax.set_xlabel("Number of Letters")
            ax.set_ylabel("Arbitrary Measurement")
            sns.scatterplot(x=sorted(y), y=np.arange(1000), hue=y, palette='husl', ax=ax)
            
            plt.show()
            
        def logistic_regression_7():
            pca = PCA(0.95)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
              
data = RobloxEDA()
