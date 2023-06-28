#####################################
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
from captum.attr import visualization as viz
from bayes_model import MnbClassifier
import numpy as np

class BaselineInterpreter():

    def __init__(self, model: MnbClassifier):
        self.model = model
        self.score_dict = self.__create_score_dict()

    
    def __transform_to_new_range(self, x, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        
        new_x = ((x - old_min) * new_range) / old_range + new_min
        return new_x


    def __create_score_dict(self):
        feature_names = self.model.tfidf.get_feature_names_out()
        true = self.model.classifier.feature_log_prob_[0]
        fake = self.model.classifier.feature_log_prob_[1]
        ratio = [(true_prob / fake_prob) for (true_prob, fake_prob) in zip(true, fake)]
        zipped = list(zip(feature_names, ratio))
        return dict(zipped)

    def interpret_text(self, text):
        text = text.lower()
        words = text.split()
        temp = [self.score_dict[word] for word in words if word in self.score_dict]
        temp = np.array(temp)
        min = np.min(temp)
        max = np.max(temp)
        neutral = (min + max) / 2

        score = [neutral if word not in self.score_dict else self.score_dict[word] for word in words]
        transformed = [self.__transform_to_new_range(x, min, max, -1, 1) for x in score]
        transformed = np.array(transformed) * -1
        return list(zip(words, transformed))


    def vizualize_interpretation(self, text, true_class, outfile="data.html", delta=0.5):
        score = self.interpret_text(text)
        predict_class = self.model.predict_text(text)[0]
        predict_proba = np.max(self.model.predict_proba_text(text)[0])

        score_vis = viz.VisualizationDataRecord(
                            word_attributions = [tup[1] for tup in score],
                            pred_prob = predict_proba,
                            pred_class = predict_class,
                            true_class = true_class,
                            attr_class = text,
                            attr_score = np.sum([tup[1] for tup in score]),
                            raw_input_ids = [tup[0] for tup in score],
                            convergence_score = delta)
        data = viz.visualize_text([score_vis])
        with open(outfile, "w") as file:
            file.write(data.data)