import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm.auto import tqdm
import os
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)

# Create directory for saving models
save_dir = "saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load AG News dataset with progress bar
print("Loading AG News dataset...")
dataset = load_dataset("ag_news")

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Tokenize datasets with progress bars
print("Tokenizing datasets...")
tokenized_train = dataset['train'].map(
    tokenize_function,
    batched=True,
    desc="Tokenizing train set"
)
tokenized_test = dataset['test'].map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test set"
)

# Convert to PyTorch format
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Create data loaders
train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True)
test_loader = DataLoader(tokenized_test, batch_size=16)

# Initialize model
print("Initializing model...")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4  # AG News has 4 classes
)

# Training settings
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop with progress tracking
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(
        dataloader,
        desc=f"Training Epoch {epoch+1}",
        leave=True,
        position=0
    )
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Update progress bar description with current loss
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

# Evaluation function with progress tracking
def evaluate(model, dataloader, device):
    model.eval()
    true_labels = []
    predictions = []
    
    progress_bar = tqdm(
        dataloader,
        desc="Evaluating",
        leave=True,
        position=0
    )
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds)
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return accuracy, f1

# Function to save model and tokenizer
def save_model(model, tokenizer, accuracy, f1, epoch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"distilbert_agnews_epoch{epoch+1}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save metrics
    with open(os.path.join(save_path, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    print(f"Model saved to {save_path}")

# Training and evaluation loop with progress tracking
print("\nStarting training...")
best_f1 = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Training
    avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    print(f"Average training loss: {avg_train_loss:.4f}")
    
    # Evaluation
    accuracy, f1 = evaluate(model, test_loader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    
    # Save model if it's the best so far
    if f1 > best_f1:
        best_f1 = f1
        save_model(model, tokenizer, accuracy, f1, epoch)

# Final evaluation
print("\nFinal Evaluation:")
accuracy, f1 = evaluate(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Save final model
save_model(model, tokenizer, accuracy, f1, num_epochs-1)
print("\nTraining completed!")