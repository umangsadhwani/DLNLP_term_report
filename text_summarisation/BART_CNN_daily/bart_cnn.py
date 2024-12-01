from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from rouge_score import rouge_scorer
import torch

# Step 1: Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Step 2: Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Step 3: Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 4: Define function to generate summary
def generate_summary(text, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Step 5: ROUGE Evaluation Function
def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    return {key: sum(value) / len(value) for key, value in rouge_scores.items()}

# Step 6: Generate summaries and evaluate on a subset of the dataset
num_samples = 100  # Number of samples to evaluate (adjust for faster results or use the full dataset)
predictions = []
references = []

for i in range(num_samples):
    article = dataset["test"][i]["article"]
    reference = dataset["test"][i]["highlights"]
    
    # Generate summary
    prediction = generate_summary(article)
    
    predictions.append(prediction)
    references.append(reference)

# Step 7: Compute ROUGE scores
rouge_scores = compute_rouge(predictions, references)

# Step 8: Output ROUGE scores
print("ROUGE Scores:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
