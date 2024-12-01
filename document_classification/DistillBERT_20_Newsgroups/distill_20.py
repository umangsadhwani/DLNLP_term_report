import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class
class NewsGroupDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, 
                                 max_length=max_length, return_tensors='pt')
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load and prepare data
def prepare_data():
    # Load 20 newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data
    labels = newsgroups.target
    
    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    return train_texts, val_texts, train_labels, val_labels

# Training function
def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            true_labels.extend(labels.numpy())
            predictions.extend(preds)
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return accuracy, f1

def main():
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=20
    )
    model.to(device)
    
    # Prepare data
    train_texts, val_texts, train_labels, val_labels = prepare_data()
    
    # Create datasets
    train_dataset = NewsGroupDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsGroupDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * 3  # 3 epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Training loop
    best_f1 = 0
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/3")
        
        # Train
        avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluate
        accuracy, f1 = evaluate(model, val_loader)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")

if __name__ == "__main__":
    main()