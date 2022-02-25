import jsonlines
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
import langdetect


class MyPreprocessor:
    def __init__(self):
        self.reviews = []
        self.voted_up = []
        self.early_access = []
        self.X = None
        self.y = None
        self.tfidf = None

    def read_data(self, path):
        print("> Reading data")
        for item in jsonlines.open(path, 'r'):
            self.reviews.append(item['text'])
            self.voted_up.append(item['voted_up'])
            self.early_access.append(item['early_access'])

    def cross_validate_min_df(self, min_df_range, max_df):
        mean_scores = []; std_scores = []
        for min_df in min_df_range:
            self.preprocess_data(predict="voted_up", min_df=min_df, max_df=max_df)
            X, y = self.get_data()
            print("\t> For min_df = %.2f..." % min_df)
            tmp_model = LogisticRegression(C=10, max_iter=1000)
            scores = cross_val_score(tmp_model, X, y, cv=5, scoring='f1')
            mean_scores.append(np.array(scores).mean())
            std_scores.append(np.array(scores).std())

        print("> min_df cross validation done")
        plt.figure()
        plt.errorbar(min_df_range, mean_scores, yerr=std_scores)
        plt.title("min_df Cross Validation")
        plt.xlabel("min_df value")
        plt.ylabel("F1-Score")
        plt.show()

    def cross_validate_max_df(self, min_df, max_df_range):
        mean_scores = []; std_scores = []
        for max_df in max_df_range:
            self.preprocess_data(predict="voted_up", min_df=min_df, max_df=max_df)
            X, y = self.get_data()
            print("\t> For max_df = %.2f..." % max_df)
            tmp_model = LogisticRegression(C=10, max_iter=1000)
            scores = cross_val_score(tmp_model, X, y, cv=5, scoring='f1')
            mean_scores.append(np.array(scores).mean())
            std_scores.append(np.array(scores).std())

        print("> max_df cross validation done")
        plt.figure()
        plt.errorbar(max_df_range, mean_scores, yerr=std_scores)
        plt.title("max_df Cross Validation")
        plt.xlabel("max_df value")
        plt.ylabel("F1-Score")
        plt.show()

    def isoToLanguage(self, lang):
        if lang == "da":
            return "danish"
        elif lang == "nl":
            return "dutch"
        elif lang == "en":
            return "english"
        elif lang == "fi":
            return "finnish"
        elif lang == "fr":
            return "french"
        elif lang == "de":
            return "german"
        elif lang == "hu":
            return "hungarian"
        elif lang == "it":
            return "italian"
        elif lang == "pt":
            return "portuguese"
        elif lang == "ro":
            return "romanian"
        elif lang == "ru":
            return "russian"
        elif lang == "es":
            return "spanish"
        elif lang == "sv":
            return "swedish"
        else:
            return None

    def tokenize(self, text, use_lang_detect=False):
        tokens = nltk.word_tokenize(text)
        stems = []

        if use_lang_detect:
            try:
                lang = langdetect.detect(text)
                lang = self.isoToLanguage(lang)
            except langdetect.lang_detect_exception.LangDetectException:
                lang = None

            for item in tokens:
                if lang:
                    stems.append(SnowballStemmer(lang).stem(item))
                else:
                    stems.append(item)
        else:
            stemmer = nltk.PorterStemmer()
            stems = [stemmer.stem(item) for item in tokens]

        return stems

    def preprocess_data(self, predict="voted_up", min_df=1, max_df=1.0):
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize, min_df=min_df, max_df=max_df, ngram_range=(1, 1))
        self.X = self.tfidf.fit_transform(self.reviews)
        self.X = self.X.toarray()

        if predict == "voted_up":
            self.y = self.voted_up
        else:
            self.y = self.early_access

    def get_data(self):
        return self.X, self.y

    def print_report(self, print_features_names=False):
        print("> Features:")
        print("\t> Number:", len(self.tfidf.get_feature_names()))
        if print_features_names:
            print("\t> Names:", self.tfidf.get_feature_names())

        print("\n> Total number of reviews:", len(self.reviews))

        voted_up_reviews = 0
        for voted_up in self.voted_up:
            if voted_up:
                voted_up_reviews += 1

        print("> Total number of \"voted_up\" reviews:", voted_up_reviews)

        early_access_reviews = 0
        for early_access in self.early_access:
            if early_access:
                early_access_reviews += 1

        print("> Total number of \"early_access\" reviews:", early_access_reviews)
