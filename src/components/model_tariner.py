from src.exception import CustomException
from src.logger import logging
import sys
import os
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from src.utlis import compute_metrics
from src.utlis import save_object
from dataclasses import dataclass
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


@dataclass
class ModelTrainerConfig:
    tarined_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,tokenized_train,tokenized_test):

        try:
            id2label = {0: "negative", 1: "positive"}
            label2id = {"negative": 0, "positive": 1}

            model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
            )

            training_args = TrainingArguments(
            output_dir="./artifacts/model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            )

            return trainer.train()


        except Exception as e:
            raise CustomException(e,sys)




