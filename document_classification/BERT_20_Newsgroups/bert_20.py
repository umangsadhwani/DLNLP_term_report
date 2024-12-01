import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

# Fetch full 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', categories=None)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(newsgroups.target)

# Tokenization and train-test split
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, labels, test_size=0.2, random_state=42
)

class NewsGroupsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BertDocClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertDocClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

def visualize_embeddings(model, dataloader, device, label_encoder):
    """Create 2D t-SNE projection of document embeddings"""
    model.eval()
    embeddings = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            bert_output = model.bert(input_ids, attention_mask)
            pooled_embedding = bert_output.pooler_output.cpu().numpy()
            
            embeddings.append(pooled_embedding)
            true_labels.extend(labels.numpy())

    embeddings = np.vstack(embeddings)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=true_labels, 
        cmap='tab20'
    )
    plt.colorbar(scatter, label='Categories')
    plt.title('t-SNE Visualization of Document Embeddings')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """Generate a heatmap of the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        rotation=90
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
NUM_CLASSES = len(newsgroups.target_names)

# Create datasets and dataloaders
train_dataset = NewsGroupsDataset(X_train, y_train, tokenizer)
test_dataset = NewsGroupsDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertDocClassifier(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1} Loss: {total_loss/len(train_loader)}')

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    correct = 0
    total = 0
    for batch in tqdm(test_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print(f'Accuracy: {100 * correct / total}%')

# Visualization and Metrics
visualize_embeddings(model, test_loader, device, label_encoder)
plot_confusion_matrix(y_true, y_pred, label_encoder)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true, 
    y_pred, 
    target_names=label_encoder.classes_
))