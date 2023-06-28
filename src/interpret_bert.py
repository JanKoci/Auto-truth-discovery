#####################################
# This file contains implementation of the BERT model interpretation
# The implementation is based on the captum library guidelines and an article by Ruben Winastwan available at 
#   https://towardsdatascience.com/interpreting-the-prediction-of-bert-model-for-text-classification-5ab09f8ef074
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
from captum.attr import LayerIntegratedGradients
import torch
from captum.attr import visualization as viz


class BertInterpreter():

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        def model_output(inputs):
            return self.model(inputs)[0]
        
        self.lig = LayerIntegratedGradients(model_output, self.model.bert.embeddings)

    def get_input_and_baseline(self, text):
        max_length = 510
        baseline_token_id = self.tokenizer.pad_token_id 
        sep_token_id = self.tokenizer.sep_token_id 
        cls_token_id = self.tokenizer.cls_token_id 

        text_ids = self.tokenizer.encode(text, truncation=True, add_special_tokens=False, max_length=max_length)
    
        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)

        baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]
        return torch.tensor([input_ids], device='cpu'), torch.tensor([baseline_input_ids], device='cpu'), token_list


    def sum_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions


    def interpret_text(self, text, true_class, outfile="data.html", n_steps=50):
        input_ids, baseline_input_ids, all_tokens = self.get_input_and_baseline(text)
        attributions, delta = self.lig.attribute(inputs = input_ids,
                                        baselines = baseline_input_ids,
                                        return_convergence_delta=True,
                                        internal_batch_size=1,
                                        target=0,
                                        n_steps=n_steps
                                        )
        attributions_sum = self.sum_attributions(attributions)

        score_vis = viz.VisualizationDataRecord(
                            word_attributions = attributions_sum,
                            pred_prob = torch.max(torch.softmax(self.model(input_ids)[0], dim=1)),
                            pred_class = torch.argmax(self.model(input_ids)[0]).numpy(),
                            true_class = true_class,
                            attr_class = text,
                            attr_score = attributions_sum.sum(),       
                            raw_input_ids = all_tokens,
                            convergence_score = delta)

        data = viz.visualize_text([score_vis])
        with open(outfile, "w") as file:
            file.write(data.data)
        print("Report written to data.html file")
    


class DistilBertInterpreter():

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        def model_output(inputs):
            return self.model(inputs)[0]
        
        self.lig = LayerIntegratedGradients(model_output, self.model.distilbert.embeddings)

    def construct_input_and_baseline(self, text):
        max_length = 510
        baseline_token_id = self.tokenizer.pad_token_id 
        sep_token_id = self.tokenizer.sep_token_id 
        cls_token_id = self.tokenizer.cls_token_id 

        text_ids = self.tokenizer.encode(text, truncation=True, add_special_tokens=False, max_length=max_length)
    
        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)

        baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]
        return torch.tensor([input_ids], device='cpu'), torch.tensor([baseline_input_ids], device='cpu'), token_list


    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions


    def interpret_text(self, text, true_class, outfile="data.html", n_steps=50):
        input_ids, baseline_input_ids, all_tokens = self.construct_input_and_baseline(text)
        attributions, delta = self.lig.attribute(inputs = input_ids,
                                        baselines = baseline_input_ids,
                                        return_convergence_delta=True,
                                        internal_batch_size=1,
                                        target=0,
                                        n_steps=n_steps
                                        )
        attributions_sum = self.summarize_attributions(attributions)

        score_vis = viz.VisualizationDataRecord(
                            word_attributions = attributions_sum,
                            pred_prob = torch.max(self.model(input_ids)[0]),
                            pred_class = torch.argmax(self.model(input_ids)[0]).numpy(),
                            true_class = true_class,
                            attr_class = text,
                            attr_score = attributions_sum.sum(),       
                            raw_input_ids = all_tokens,
                            convergence_score = delta)

        data = viz.visualize_text([score_vis])
        with open(outfile, "w") as file:
            file.write(data.data)
        print("Report written to data.html file")
        return data