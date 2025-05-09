# bert_kg.py

import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.kg.graph_creator import GraphCreator, GraphRelation

class BERT_KG(GraphCreator):
    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def create_graph_relations(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=10,
                early_stopping=False,
            )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                
        # Clean and parse output
        cleaned_output = re.sub(r"<s>|</s>", "", decoded_output).strip()
        triplet_info = cleaned_output.split("<triplet> ")[1:]

        triplets = []
        for entry in triplet_info:
            split_version = entry.split("<subj> ")
            subject = split_version[0]
            for obj_rel in split_version[1:]:
                if "<obj>" not in obj_rel:
                    continue
                try:
                    obj, rel = obj_rel.split("<obj> ")[:2]
                    triplets.append(GraphRelation(subject.strip(), obj.strip(), rel.strip()))
                except ValueError:
                    continue  # Malformed triple part

        return triplets