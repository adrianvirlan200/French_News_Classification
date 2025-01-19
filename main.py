import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
import torch
from torch.utils.data import Dataset
import string

import nltk
nltk.download('stopwords')

# Load datasets
df = pd.read_csv('train.csv', index_col=False)
test_data = pd.read_csv('test.csv', index_col=False)

# Set a random seed for reproducibility
random_seed = random.randint(0, 100000)
print(random_seed)

# 49119

# random_seed = 6186
# random_seed = 42

# Split the data into train and validation sets
train_df, validation_df = train_test_split(
    df,
    test_size=0.1,
    random_state=random_seed,
    shuffle=True,
    stratify=df['Label']
)

fake_df = train_df[train_df['Label'] == 'fake']
biased_df = train_df[train_df['Label'] == 'biased']
true_df = train_df[train_df['Label'] == 'true']

max_count = max(len(fake_df), len(biased_df), len(true_df))

fake_upsampled = fake_df.sample(
    n=max_count, replace=True, random_state=random_seed)
biased_upsampled = biased_df.sample(
    n=max_count, replace=True, random_state=random_seed)
true_upsampled = true_df.sample(
    n=max_count, replace=True, random_state=random_seed)

train_df_balanced = pd.concat(
    [fake_upsampled, biased_upsampled, true_upsampled], ignore_index=True)
# Amestecă rândurile (shuffle)
train_df_balanced = train_df_balanced.sample(
    frac=1, random_state=random_seed).reset_index(drop=True)

# Preprocessing function

# 0.7384615384615385
# deleted > acc
# lem, pct -> 0.7743589743589744
# lem, pct, upper -> 0.74
# lem, STOP, pct -> 0.

# lem, pct + 7 epoch -> 0.81


def processData(df):
    # Lowercase the text
    df['Text'] = df['Text'].str.lower()
    # Remove punctuation
    # df['Text'] = df['Text'].str.translate(
    #     str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop = set(stopwords.words('french'))
    df['Text'] = df['Text'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in stop]))

    # Stemming/Lemmatization
    # stemmer = SnowballStemmer('french')
    # df['Text'] = df['Text'].apply(lambda x: ' '.join(
    #     [stemmer.stem(word) for word in x.split()]))

    return df


# Process datasets
train_data = processData(train_df_balanced)
validation_data = processData(validation_df)
test_data = processData(test_data)

# Define Dataset class


class T5ClassificationDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx].clone().detach()
        return item


# Initialize tokenizer and model
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Format input text


def format_input(text):
    return f'classify this news article: {text}'


# Apply formatting to train and validation data
train_inputs = train_data['Text'].apply(format_input).tolist()
train_targets = train_data['Label'].tolist()
validation_inputs = validation_data['Text'].apply(format_input).tolist()
validation_targets = validation_data['Label'].tolist()
test_inputs = test_data['Text'].apply(
    lambda x: f'classify this news article: {x}').tolist()

# Tokenize inputs
train_encodings = tokenizer(
    train_inputs,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors='pt'
)
validation_encodings = tokenizer(
    validation_inputs,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors='pt'
)
test_encodings = tokenizer(
    test_inputs,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors='pt'
)

# Create datasets
train_dataset = T5ClassificationDataset(train_encodings, tokenizer(
    train_targets, padding=True, truncation=True, max_length=10, return_tensors='pt')['input_ids'])
validation_dataset = T5ClassificationDataset(validation_encodings, tokenizer(
    validation_targets, padding=True, truncation=True, max_length=10, return_tensors='pt')['input_ids'])
test_dataset = T5ClassificationDataset(test_encodings)

# Define compute_metrics function


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = logits.argmax(axis=-1)
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {'accuracy': accuracy_score(decoded_labels, decoded_preds)}


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=7,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    fp16=True,
    dataloader_num_workers=4
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

final_metrics = trainer.evaluate()  # This uses the validation dataset by default
print("Final Validation Results:", final_metrics)
print("Final Validation Accuracy:", final_metrics["eval_accuracy"])

# Save the model
trainer.save_model('./t5-base-finetuned-classification')
tokenizer.save_pretrained('./t5-base-finetuned-classification')

# Generate predictions for test set
model.eval()

batch_size = 16
all_predictions = []
for i in range(0, len(test_dataset), batch_size):
    input_ids_batch = test_encodings['input_ids'][i:i+batch_size].to(device)
    attention_mask_batch = test_encodings['attention_mask'][i:i +
                                                            batch_size].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_predictions.extend(decoded)

# Convert to final list
decoded_test_preds = [pred.strip() for pred in all_predictions]

original_test_data = pd.read_csv('test.csv', index_col=False)
original_test_data['Predicted_Label'] = decoded_test_preds
original_test_data.to_csv('classification_result.csv', index=False)
