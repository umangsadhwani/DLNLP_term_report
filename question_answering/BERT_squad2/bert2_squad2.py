import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadV1Processor, squad_convert_examples_to_features
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_dataset

# Load pre-trained BERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load SQuAD dataset
processor = SquadV1Processor()
# Load SQuAD dataset using the datasets library
squad_dataset = load_dataset('squad')

train_examples = squad_dataset['train']
eval_examples = squad_dataset['validation']
# eval_examples = processor.get_dev_examples('path_to_squad_data')

# Convert examples to features
train_features = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset='pt'
)

eval_features = squad_convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset='pt'
)

# Create DataLoader
train_dataloader = DataLoader(train_features, sampler=RandomSampler(train_features), batch_size=8)
eval_dataloader = DataLoader(eval_features, sampler=SequentialSampler(eval_features), batch_size=8)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_dataloader) * 3  # Number of training epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):  # Number of training epochs
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        optimizer.zero_grad()
        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'start_positions': batch[3].to(device),
            'end_positions': batch[4].to(device)
        }
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': epoch_loss / len(progress_bar)})

    print(f"Epoch {epoch + 1} loss: {epoch_loss / len(train_dataloader)}")

# Evaluation
model.eval()
all_predictions = []
all_true_starts = []
all_true_ends = []

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    with torch.no_grad():
        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device)
        }
        outputs = model(**inputs)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        all_predictions.append((start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)))
        all_true_starts.append(batch[3])
        all_true_ends.append(batch[4])

# Calculate EM and F1
def calculate_em_f1(predictions, true_starts, true_ends):
    em = 0
    f1 = 0
    for pred, true_start, true_end in zip(predictions, true_starts, true_ends):
        pred_start, pred_end = pred
        true_start, true_end = true_start.item(), true_end.item()
        if pred_start == true_start and pred_end == true_end:
            em += 1
        pred_tokens = set(range(pred_start, pred_end + 1))
        true_tokens = set(range(true_start, true_end + 1))
        common_tokens = pred_tokens & true_tokens
        if len(common_tokens) == 0:
            f1 += 0
        else:
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(true_tokens)
            f1 += 2 * (precision * recall) / (precision + recall)
    em = em / len(predictions)
    f1 = f1 / len(predictions)
    return em, f1

em, f1 = calculate_em_f1(all_predictions, all_true_starts, all_true_ends)
print(f"Exact Match (EM): {em}")
print(f"F1 Score: {f1}")