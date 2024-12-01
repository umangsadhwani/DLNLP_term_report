import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from evaluate import load
import numpy as np
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset Class for T5
class XSumDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Prepare input with T5's specific format
        input_text = f"summarize: {self.data[idx]['document']}"
        
        # Tokenize input document
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_input_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )

        # Tokenize summary
        targets = self.tokenizer(
            self.data[idx]['summary'], 
            max_length=self.max_target_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Training Function
def train(model, train_loader, val_loader, optimizer, device, epochs=3):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss
                total_val_loss += loss.item()

        # Print epoch statistics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './best_t5_xum_model.pth')

    return model

# Evaluation Function
def evaluate_model(model, val_loader, tokenizer, device):
    # Load evaluation metrics
    rouge_scorer = load('rouge')
    bert_scorer = load('bertscore')

    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            # Generate summaries
            # T5 specific generation with prefix
            generated_ids = model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True
            )

            # Decode predictions and references
            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(pred_texts)
            references.extend(ref_texts)

    # Compute ROUGE scores
    rouge_results = rouge_scorer.compute(
        predictions=predictions, 
        references=references
    )

    # Compute BERT scores
    bert_results = bert_scorer.compute(
        predictions=predictions, 
        references=references, 
        lang='en'
    )

    # Aggregate metrics
    metrics = {
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        'bert_precision': np.mean(bert_results['precision']),
        'bert_recall': np.mean(bert_results['recall']),
        'bert_f1': np.mean(bert_results['f1'])
    }

    return metrics

def main():
    # Set up device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset('xsum')
    
    # Create datasets
    train_dataset = dataset['train'].shuffle(seed=42).select(range(20000))
    val_dataset = dataset['validation'].shuffle(seed=42).select(range(10000))

    # Create data loaders
    train_data = XSumDataset(train_dataset, tokenizer)
    val_data = XSumDataset(val_dataset, tokenizer)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Train the model
    trained_model = train(model, train_loader, val_loader, optimizer, device)

    # Evaluate the model
    metrics = evaluate_model(trained_model, val_loader, tokenizer, device)

    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()