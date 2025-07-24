from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Value

@dataclass
class DataTransformationConfig:
    preprocessor_obg_file=os.path.join('artifacts','proprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transform_config=DataTransformationConfig()

    def get_data_transform_obg(self):

        try:
            tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
            return tokenizer

        except Exception as e:
            raise CustomException(e,sys)
        
    def preprocess_function(self,examples):

        try:
            tokenizer = self.get_data_transform_obg()
            tokenizer_output = tokenizer(examples["review"], truncation=True)
            return tokenizer_output

        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def label_to_int(example):
        return {"label": 1 if example["label"].lower() == "positive" else 0}
        
    def initiate_data_transform(self,train_path,test_path):

        try:
            train_files=load_dataset("csv", data_files=train_path)["train"]
            test_files=load_dataset("csv", data_files=test_path)["train"]

            train_files = train_files.rename_column("sentiment", "label")
            test_files = test_files.rename_column("sentiment", "label")

            train_files=train_files.map(self.label_to_int)
            test_files=test_files.map(self.label_to_int)

            logging.info("Read train and test data completed")


            logging.info("calling the data transform object")


            tokenized_train=train_files.map(self.preprocess_function,batched=True)

            tokenized_test=test_files.map(self.preprocess_function,batched=True)

            logging.info("Tokenized both train and test data")

            
            return(
               tokenized_train,tokenized_test
            )

        except Exception as e:
            raise CustomException(e,sys)
            
            


