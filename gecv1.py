import torch
import pandas as pd
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import logging
logging.basicConfig(
filename="step.log",
encoding="utf-8",
filemode="a",
format="{asctime} - {levelname} - {message}",
style="{",
datefmt="%Y-%m-%d %H:%M",
)
logging.warning("import done")

df = pd.read_csv('/content/Cleaned_Lang8.csv')
# df.shape
df.rename(columns={'0': 'Input', '1': 'Output'}, inplace=True)
logging.warning("csv read successfully")

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.10, shuffle=True)
train_df.shape, test_df.shape

def calc_token_len(example):
    return len(tokenizer(example).input_ids)

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

logging.warning("model instantiated")

test_df['input_token_len'] = test_df['Input'].apply(calc_token_len)
# test_df.head()
test_df['input_token_len'].describe()

from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

from torch.utils.data import Dataset, DataLoader
class GrammarDataset(Dataset):
    def __init__(self, dataset, tokenizer,print_text=False):
        self.dataset = dataset
        self.pad_to_max_length = False
        self.tokenizer = tokenizer
        self.print_text = print_text
        self.max_len = 64

    def __len__(self):
        return len(self.dataset)


    def tokenize_data(self, example):
        input_, target_ = example['Input'], example['Output']

        # tokenize inputs
        tokenized_inputs = tokenizer(input_, padding='max_length',
                                            max_length=self.max_len,
                                            return_attention_mask=True, truncation=True)

        tokenized_targets = tokenizer(target_, padding='max_length',
                                            max_length=self.max_len,
                                            return_attention_mask=True, truncation=True)

        inputs={"input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask'],
            "labels": tokenized_targets['input_ids']
        }

        return inputs


    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        if self.print_text:
            for k in inputs.keys():
                print(k, len(inputs[k]))

        return inputs


BATCH_SIZE = 16
MAX_LENGTH = 256
EPOCHS = 1
NUM_WORKERS = 2
OUT_DIR = 'results_t5_base'

# Define the training arguments
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=OUT_DIR,
    eval_strategy='steps',
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to='tensorboard'
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=GrammarDataset(train_dataset, tokenizer),
    eval_dataset=GrammarDataset(test_dataset, tokenizer),
)

logging.warning("training started")

trainer.train()

logging.warning("training ended")

tokenizer.save_pretrained('t5_um_gec_model')

logging.warning("tokenizer saved")

trainer.save_model('t5_um_gec_model')

logging.warning("model saved")

