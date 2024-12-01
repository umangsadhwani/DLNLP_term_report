import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm.auto import tqdm  # Import tqdm for progress bars
import collections  # Import collections module
import re  # Import re module for regular expressions

# Device configuration
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def normalize_answer(s):
    """
    Normalize answer for evaluation
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """
    Check if prediction exactly matches ground truth
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class SQuADDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        context = self.dataset[idx]['context']
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answers']['text'][0]  # Take first answer

        input_text = f"question: {question} context: {context}"
        target_text = answer

        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        targets = self.tokenizer(
            target_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'original_answer': answer,
            'context': context,
            'question': question
        }

def evaluate_model(model, dataloader, tokenizer, device):
    """
    Evaluate model performance with EM and F1 scores
    """
    model.eval()
    em_scores = []
    f1_scores = []
    
    all_em_probs = []
    all_f1_probs = []
    
    # Wrap dataloader with tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_answers = batch['original_answer']
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=50
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Calculate metrics
            for pred, true_answer in zip(predictions, original_answers):
                em = exact_match_score(pred, true_answer)
                f1 = f1_score(pred, true_answer)
                
                em_scores.append(1 if em else 0)
                f1_scores.append(f1)
                
                # For PR curve
                all_em_probs.append(1 if em else 0)
                all_f1_probs.append(f1)
            
            # Update progress bar description with current metrics
            progress_bar.set_postfix({
                'EM': np.mean(em_scores),
                'F1': np.mean(f1_scores)
            })
    
    # Calculate average metrics
    avg_em = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)
    
    # Plot Precision-Recall Curves
    plot_pr_curves(all_em_probs, all_f1_probs)
    
    return avg_em, avg_f1

def plot_pr_curves(em_scores, f1_scores):
    """
    Plot Precision-Recall curves for EM and F1 scores
    """
    plt.figure(figsize=(12, 5))
    
    # EM Curve
    plt.subplot(1, 2, 1)
    precision_em, recall_em, _ = precision_recall_curve(em_scores, em_scores)
    avg_precision_em = average_precision_score(em_scores, em_scores)
    plt.plot(recall_em, precision_em, label=f'EM (AP = {avg_precision_em:.2f})')
    plt.title('Precision-Recall Curve (Exact Match)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    # F1 Score Curve
    plt.subplot(1, 2, 2)
    f1_labels = [1 if score > 0.5 else 0 for score in f1_scores]
    precision_f1, recall_f1, _ = precision_recall_curve(f1_labels, f1_scores)
    avg_precision_f1 = average_precision_score(f1_labels, f1_scores)
    plt.plot(recall_f1, precision_f1, label=f'F1 (AP = {avg_precision_f1:.2f})')
    plt.title('Precision-Recall Curve (F1 Score)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('squad_pr_curves.png')
    plt.close()

def train_model(model, train_dataloader, val_dataloader, tokenizer, optimizer, scheduler, device, epochs=3):
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        # Wrap train_dataloader with tqdm for progress bar
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for batch in train_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar description with current loss
            train_progress_bar.set_postfix({
                'Loss': loss.item()
            })
        
        # Validation phase
        avg_em, avg_f1 = evaluate_model(model, val_dataloader, tokenizer, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {total_train_loss/len(train_dataloader):.4f}")
        print(f"Validation Exact Match: {avg_em:.4f}")
        print(f"Validation F1 Score: {avg_f1:.4f}")
        print("-" * 50)

def main():
    # Necessary imports for evaluation
    import re
    import collections
    
    # Load pre-trained T5 model and tokenizer
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load SQuAD dataset
    squad_dataset = load_dataset('squad')
    train_dataset = squad_dataset['train']
    val_dataset = squad_dataset['validation']

    # Create custom datasets
    train_dataset = SQuADDataset(train_dataset, tokenizer)
    val_dataset = SQuADDataset(val_dataset, tokenizer)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Prepare optimizer and schedule
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )

    # Train the model
    train_model(model, train_dataloader, val_dataloader, tokenizer, optimizer, scheduler, device)

    # Final evaluation
    final_em, final_f1 = evaluate_model(model, val_dataloader, tokenizer, device)
    print("\nFinal Evaluation:")
    print(f"Exact Match Score: {final_em:.4f}")
    print(f"F1 Score: {final_f1:.4f}")

    # Save the model
    model.save_pretrained('./t5_squad_finetuned')
    tokenizer.save_pretrained('./t5_squad_finetuned')

if __name__ == "__main__":
    main()