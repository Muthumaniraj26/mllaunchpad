import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import os

# --- 1. The Dataset (Included directly in the script) ---
# This removes the need for an external tasks_dataset.csv file.
data = [
    {"text": "I need to figure out if a movie review is positive or negative.", "label": "text classification"},
    {"text": "Classify these customer support tickets into 'urgent', 'billing', or 'technical'.", "label": "text classification"},
    {"text": "Is this news headline about sports, politics, or technology?", "label": "text classification"},
    {"text": "Detect if a comment is toxic or not.", "label": "text classification"},
    {"text": "Given a person's age and income, predict if they will buy a product.", "label": "text classification"},
    {"text": "Predict the price of a house based on its size and number of rooms.", "label": "tabular regression"},
    {"text": "Forecast next month's sales based on past performance and ad spend.", "label": "tabular regression"},
    {"text": "Estimate the total power consumption for a city tomorrow.", "label": "tabular regression"},
    {"text": "From a patient's health metrics, estimate their insurance cost.", "label": "tabular regression"},
    {"text": "Given a stock's history, predict its price for next week.", "label": "tabular regression"},
    {"text": "Write a short, creative story about a robot exploring a new planet.", "label": "text generation"},
    {"text": "Generate a marketing email to announce a new product launch.", "label": "text generation"},
    {"text": "Create a poem about the monsoon season in India.", "label": "text generation"},
    {"text": "Compose a professional thank you note to a client.", "label": "text generation"},
    {"text": "Write a Python function that takes a list and returns the sum.", "label": "text generation"},
    {"text": "Summarize this long news article into three main bullet points.", "label": "summarization"},
    {"text": "Condense this research paper's abstract into a single sentence.", "label": "summarization"},
    {"text": "Create a short summary of a long business meeting transcript.", "label": "summarization"},
    {"text": "Shorten this chapter of a book into a one-paragraph summary.", "label": "summarization"},
    {"text": "Give me the key takeaways from this legal document.", "label": "summarization"},
    {"text": "Based on this article about the history of Rajapalayam, what is the town known for?", "label": "question answering"},
    {"text": "From this product manual, how do I reset the device to factory settings?", "label": "question answering"},
    {"text": "What was the main cause of the event described in this historical text?", "label": "question answering"},
    {"text": "Who was the main character in the first chapter of this story?", "label": "question answering"},
    {"text": "According to the report, what were the quarterly earnings?", "label": "question answering"},
]

# --- 2. Prepare and Process the Data ---
df = pd.DataFrame(data)

# Create a mapping from text labels to integer IDs
# Create a mapping from text labels to integer IDs
labels = df['label'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# Create the new column with the integer IDs
df['label_id'] = df['label'].map(label2id).astype(int)  # Ensure label_id is an integer

# Split data
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Rename 'label_id' to 'labels' for Trainer compatibility BEFORE tokenization
train_dataset = train_dataset.rename_column("label_id", "labels")
eval_dataset = eval_dataset.rename_column("label_id", "labels")

# --- 3. Tokenize the Data ---
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)


# --- 4. Load Model and Define Training Arguments ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./training_output",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# --- 5. Train and Save the Model ---
print("\n🚀 Starting model fine-tuning...")
trainer.train()
print("✅ Model fine-tuning complete!")

# --- 6. Save the Final Model ---
final_model_path = "./my_custom_task_classifier"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"✅ Model saved successfully to the '{final_model_path}' folder.")