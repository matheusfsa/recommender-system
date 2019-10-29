import pandas as pd
import numpy as np
# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error


class IMDBContentBased:

    def __init__(self):
        self.df_imdb = pd.read_csv('./datasets/IMDB-Movie-Data.csv')
        self.df_imdb.index = self.df_imdb.Title
        self.recommendations = pd.DataFrame()
        self.profile = None
        try:
            self.X = pd.read_csv('./datasets/X.csv', index_col=0)
            self.y = pd.read_csv('./datasets/y.csv', index_col=0)
            try:
                self.y = self.y.drop(columns="Title").iloc[:50,:]
            except KeyError:
                pass
            self.X_train = pd.read_csv('./datasets/X_train.csv', index_col=0).iloc[:50,:]
        except FileNotFoundError:
            self.X = pd.DataFrame()
            self.y = pd.DataFrame()
            self.X_train = pd.DataFrame()
            self.item_description()
    # Most frequently occuring words

    def get_top_n_words(corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq[:n]

    def tf_id(self, column, max_features,n_min, n_max):
        df_imdb = self.df_imdb.copy()
        df_imdb["Word_count"] = df_imdb[column].apply(lambda x: len(str(x).split(" ")))
        stop_words = set(stopwords.words("english"))
        corpus = []
        for i in range(0, df_imdb.shape[0]):
            # Remove punctuations
            text = re.sub('[^a-zA-Z]', ' ', df_imdb[column][i])

            # Convert to lowercase
            text = text.lower()

            # remove tags
            text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

            # remove special characters and digits
            text = re.sub("(\\d|\\W)+", " ", text)

            ##Convert to list from string
            text = text.split()

            ##Stemming
            ps = PorterStemmer()
            # Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text if not word in
                                                                stop_words]
            text = " ".join(text)
            corpus.append(text)
        cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=max_features, ngram_range=(n_min, n_max))
        X = cv.fit_transform(corpus)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        tf_idf_X = tfidf_transformer.transform(X)
        return pd.DataFrame(tf_idf_X.toarray(), index=df_imdb.Title, columns=cv.get_feature_names())

    def item_description(self):
        df_imdb = self.df_imdb.copy()
        desc_feats = self.tf_id("Description", 1000,1,  3)
        self.X[desc_feats.columns] = desc_feats
        directors_feats = pd.get_dummies(df_imdb['Director'])*0.4
        directors_feats.index = df_imdb.Title
        df_imdb.index = df_imdb.Title
        self.X["Runtime (Minutes)"] = ((df_imdb["Runtime (Minutes)"].max() - df_imdb["Runtime (Minutes)"]) / (df_imdb["Runtime (Minutes)"].max() - df_imdb["Runtime (Minutes)"].min()))*0.3
        self.X["Year"] = ((df_imdb["Year"].max() - df_imdb["Year"]) / (df_imdb["Year"].max() - df_imdb["Year"].min()))*0.3
        self.X["Rating"] = df_imdb["Rating"]*0.1
        title_feats = self.tf_id("Title", 100, 1, 3)
        self.X[title_feats.columns] = title_feats
        actors_feats = self.tf_id("Actors", 1000, 2, 2)
        self.X[actors_feats.columns] = actors_feats
        genre_feats = self.tf_id("Genre", 20, 1, 1)
        self.X[genre_feats.columns] = genre_feats

    def initial_evaluations(self, n_samples=5):
        sample = self.X.sample(n_samples)
        self.collect_evaluations(sample, n_samples)

    def collect_evaluations(self, sample, n_samples=5):
        y = self.y.values
        for filme in sample.index:
            nota = np.float(input("Qual a sua nota para o filme " + str(filme) + "?"))
            if nota != -1:
                y = np.append(y, nota)
                self.X = self.X.drop(index=filme)
            else:
                sample = sample.drop(index=filme)
        self.y = pd.DataFrame(y)
        self.X_train = self.X_train.append(sample)
        self.save_train_data()

    def save_train_data(self):
        self.X_train.to_csv('/datasets/X_train.csv')
        self.y['Title'] = self.X_train.index
        self.y.to_csv('./datasets/y.csv')
        self.X.to_csv('./datasets/X.csv')
        self.y = self.y.drop(columns="Title")

    def profile_learning(self):
        self.profile = LassoCV(alphas=[0.05, 0.25, 0.5, 0.75, 1, 5, 10, 30], cv=5, random_state=0).fit(self.X_train,
                                                                                              self.y.values.reshape(
                                                                                                  -1, ))

    def filtering(self,  n=10):
        self.recommendations = pd.DataFrame(self.profile.predict(self.X), index=self.X.index)[0].sort_values(ascending=False)[1:n]
        return self.recommendations

    def feedback(self):
        self.collect_evaluations(self.X.loc[self.recommendations.index, :], self.recommendations.shape[0])



