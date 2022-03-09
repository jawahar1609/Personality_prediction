# importing dependencies here
import numpy as np
import pandas as pd
import os

import regex as re
import nltk

# lemmatizing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# sentiment scoring
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# pos tagging
from nltk.tokenize import word_tokenize

# importing model
from joblib import load

class PersonalityPrediction():

    def __init(self):
        pass

    def unique_words(self, s):
        unique = set(s.split(" "))
        return len(unique)

    def emojis(self, post):
        emoji_count = len(re.findall(r'[\U0001f600-\U0001f650]', post))
        return emoji_count

    def colons(self, post):
        colon_count = 0
        words = post.split()
        for e in words:
            if "http" not in e:
                colon_count += e.count(":")
        return colon_count

    def lemmatize(self, s):
        lemmatizer = WordNetLemmatizer()
        new_s = ""
        for word in s.split(" "):
            lemmatizer.lemmatize(word)
            if word not in stopwords.words("english"):
                new_s += word + " "
        return new_s[:-1]

    def clean(self, s):
        mbti = ["INFP","INFJ","INTP","INTJ","ENTP","ENFP","ISTP","ISFP","ENTJ","ISTJ","ENFJ","ISFJ","ESTP","ESFP","ESFJ","ESTJ",]
        # remove urls
        s = re.sub(re.compile(r"https?:\/\/(www)?.?([A-Za-z_0-9-]+).*"), "", s)
        # remove emails
        s = re.sub(re.compile(r"\S+@\S+"), "", s)
        # remove punctuation
        s = re.sub(re.compile(r"[^a-z\s]"), "", s)
        # Make everything lowercase
        s = s.lower()
        # remove all personality types
        for type_word in mbti:
            s = s.replace(type_word.lower(), "")
        
        return s


    def prep_counts(self, s):
        clean_s = self.clean(s)
        d = {
            "clean_posts": self.lemmatize(clean_s),
            "link_count": s.count("http"),
            "youtube": s.count("youtube") + s.count("youtu.be"),
            "img_count": len(re.findall(r"(\.jpg)|(\.jpeg)|(\.gif)|(\.png)", s)),
            "upper": len([x for x in s.split() if x.isupper()]),
            "char_count": len(s),
            "word_count": clean_s.count(" ") + 1,
            "qm": s.count("?"),
            "em": s.count("!"),
            "colons": self.colons(s),
            "emojis": self.emojis(s),
            "unique_words": self.unique_words(clean_s),
            "ellipses": len(re.findall(r"\.\.\.\ ", s)),
        }
        return clean_s, d


    def prep_sentiment(self, s):
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(s)
        d = {
            "compound_sentiment": score["compound"],
            "pos_sentiment": score["pos"],
            "neg_sentiment": score["neg"],
            "neu_sentiment": score["neu"],
        }
        return d


    def tag_pos(self, s):
        tagged_words = nltk.pos_tag(word_tokenize(s))
        tags_dict = {
            "ADJ_avg": ["JJ", "JJR", "JJS"],
            "ADP_avg": ["EX", "TO"],
            "ADV_avg": ["RB", "RBR", "RBS", "WRB"],
            "CONJ_avg": ["CC", "IN"],
            "DET_avg": ["DT", "PDT", "WDT"],
            "NOUN_avg": ["NN", "NNS", "NNP", "NNPS"],
            "NUM_avg": ["CD"],
            "PRT_avg": ["RP"],
            "PRON_avg": ["PRP", "PRP$", "WP", "WP$"],
            "VERB_avg": ["MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ".": ["#", "$", "''", "(", ")", ",", ".", ":"],
            "X": ["FW", "LS", "UH"],
        }
        d = dict.fromkeys(tags_dict, 0)
        for tup in tagged_words:
            tag = tup[1]
            for key, val in tags_dict.items():
                if tag in val:
                    tag = key
            d[tag] += 1
        return d


    def prep_data(self, s):
        clean_s, d = self.prep_counts(s)
        sentiment = self.prep_sentiment(self.lemmatize(clean_s))
        d.update(sentiment)
        d.update(self.tag_pos(clean_s))
        features = [
            "clean_posts",
            "compound_sentiment",
            "ADJ_avg",
            "ADP_avg",
            "ADV_avg",
            "CONJ_avg",
            "DET_avg",
            "NOUN_avg",
            "NUM_avg",
            "PRT_avg",
            "PRON_avg",
            "VERB_avg",
            "qm",
            "em",
            "colons",
            "emojis",
            "word_count",
            "unique_words",
            "upper",
            "link_count",
            "ellipses",
            "img_count",
        ]

        return pd.DataFrame([d])[features], sentiment

    def combine_classes(self, y_pred1, y_pred2, y_pred3, y_pred4):

        combined = []
        for i in range(len(y_pred1)):
            combined.append(str(y_pred1[i]) + str(y_pred2[i]) + str(y_pred3[i]) + str(y_pred4[i]))

        type_list = [
        {"0": "I", "1": "E"},
        {"0": "N", "1": "S"},
        {"0": "F", "1": "T"},
        {"0": "P", "1": "J"},
        ]
        
        type_dict = {"I":"Introversion", "E":"Extroversion", "S": "Sensing", "N":"Intuition",
                    "F":"Feeling", "T":"Thinking", "P":"Perceiving", "J":"Judging"}
        result = []
        for num in combined:
            s = ""
            lt = []
            for i in range(len(num)):
                res = type_list[i][num[i]]
                lt.append(type_dict[res])
                s += res
            result.append(s)
            result.append(lt)

        return result

    def predict(self, s):

        s = s['text']
        if s == "":
            return "No text" 

        X, output = self.prep_data(s)

        # loading the 4 models
        EorI_model = load("app/trained_weights/Extroversion.joblib")
        SorN_model = load("app/trained_weights/Sensing.joblib")
        TorF_model = load("app/trained_weights/Thinking.joblib")
        JorP_model = load("app/trained_weights/Judging.joblib")

        # predicting
        EorI_pred = EorI_model.predict(X)
        SorN_pred = SorN_model.predict(X)
        TorF_pred = TorF_model.predict(X)
        JorP_pred = JorP_model.predict(X)

        # combining the predictions from the 4 models
        result = self.combine_classes(EorI_pred, SorN_pred, TorF_pred, JorP_pred)

        for key in output.keys():
            output[key]= round(output[key]*100, 2)
        output["result"] = result[0]
        output["typeExp"] = " - ".join(result[1])
        
        return output