import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from collections import Counter
import string
import re
import evaluate
import time
from datetime import timedelta

class SQuAD2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"question: {item['question']} context: {item['context']}"
        answer_text = ""
        if len(item['answers']['text']) > 0:
            answer_text = item['answers']['text'][0]
        else:
            answer_text = "no answer"

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                answer_text,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'target_text': answer_text
        }

def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def format_time(elapsed):
    return str(timedelta(seconds=int(elapsed)))

def train_model():
    print("Loading dataset...")
    dataset = load_dataset("squad_v2")
    
    print("Initializing tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    print("Preparing datasets...")
    with tqdm(desc="Creating train dataset", total=1) as pbar:
        train_dataset = SQuAD2Dataset(dataset['train'], tokenizer)
        pbar.update(1)
    
    with tqdm(desc="Creating validation dataset", total=1) as pbar:
        eval_dataset = SQuAD2Dataset(dataset['validation'], tokenizer)
        pbar.update(1)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0)
    total_start_time = time.time()
    
    best_f1 = 0
    best_em = 0
    
    print("\nStarting training...")
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        # Progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=1, leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            labels[labels == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = format_time(time.time() - epoch_start_time)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Epoch time: {epoch_time}")
        
        # Evaluation
        print("\nStarting evaluation...")
        model.eval()
        exact_matches = []
        f1_scores = []
        
        eval_pbar = tqdm(eval_loader, desc="Evaluating", position=1, leave=False)
        
        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_texts = batch['target_text']
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
                
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for pred, target in zip(predictions, target_texts):
                    em_score = compute_exact_match(pred, target)
                    f1_score = compute_f1(pred, target)
                    exact_matches.append(em_score)
                    f1_scores.append(f1_score)
                
                # Update evaluation progress bar
                current_em = np.mean(exact_matches)
                current_f1 = np.mean(f1_scores)
                eval_pbar.set_postfix({
                    'EM': f'{current_em:.4f}',
                    'F1': f'{current_f1:.4f}'
                })
        
        avg_em = np.mean(exact_matches)
        avg_f1 = np.mean(f1_scores)
        
        # Update best scores
        best_em = max(best_em, avg_em)
        best_f1 = max(best_f1, avg_f1)
        
        print(f"\nEvaluation Results - Epoch {epoch + 1}:")
        print(f"Exact Match: {avg_em:.4f} (Best: {best_em:.4f})")
        print(f"F1 Score: {avg_f1:.4f} (Best: {best_f1:.4f})")
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_train_loss:.4f}',
            'EM': f'{avg_em:.4f}',
            'F1': f'{avg_f1:.4f}'
        })
    
    total_time = format_time(time.time() - total_start_time)
    print(f"\nTraining completed! Total time: {total_time}")
    print(f"Best Exact Match: {best_em:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    train_model()