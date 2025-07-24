import pandas as pd
from transformers import pipeline
from src.exception import CustomException
from src.logger import logging
import sys
import os

class SentimentPredictor:
    def __init__(self, model_name_or_path="path to the model.pkl"):

        try:
            self.classifier = pipeline("sentiment-analysis", model=model_name_or_path)

        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_text(self, text: str):

        try:
            result = self.classifier(text)
            return result
        
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_file(self, input_csv_path: str, output_csv_path: str, text_column: str = "review"):

        try:
    
            df = pd.read_csv(input_csv_path)

            if text_column not in df.columns:
                raise CustomException(f"'{text_column}' column not found in the input file.", sys)

            predictions = self.classifier(df[text_column].tolist(), truncation=True)

            df["predicted_label"] = [pred["label"] for pred in predictions]
            df["confidence_score"] = [pred["score"] for pred in predictions]

            df.to_csv(output_csv_path, index=False)

            return output_csv_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        input_csv = "data/test.csv"       
        output_csv = "artifacts/predictions.csv"  
        text_col = "review"                

        predictor = SentimentPredictor()
        result_path = predictor.predict_from_file(input_csv, output_csv, text_column=text_col)

        print(f"Predictions saved to: {result_path}")

    except Exception as e:
        raise CustomException(e, sys)
