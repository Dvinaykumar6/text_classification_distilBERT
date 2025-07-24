import evaluate
import numpy as np
from src.exception import CustomException
import sys
import os
import pickle
import dill
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

accuracy=accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    except Exception as e:
        raise CustomException(e,sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def tokenizer_obj():
    try:
        tokenizer=AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        return tokenizer
    
    except Exception as e:
        raise CustomException(e,sys)
    
def data_collector():
    try:
        datacollector= DataCollatorWithPadding(tokenizer=tokenizer_obj)
        
    except Exception as e:
        raise CustomException(e,sys)