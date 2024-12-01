import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Step 1: Custom Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        Initialize the dataset with tokenized encodings and labels.
        :param encodings: Dictionary of tokenized inputs (input_ids, attention_mask, token_type_ids).
        :param labels: List of labels corresponding to the inputs.
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset as a dictionary of tensors.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = torch.tensor(self.labels[idx])
        return item, label

# Step 2: Preprocess Data
def preprocess_data(dataset_name="ag_news", model_name="bert-base-uncased", max_length=128):
    """
    Load and preprocess the dataset.
    :param dataset_name: Name of the dataset (e.g., 'ag_news').
    :param model_name: Name of the pre-trained model (e.g., 'bert-base-uncased').
    :param max_length: Maximum token length for truncation/padding.
    :return: Train and test datasets wrapped in the custom TextDataset class.
    """
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    print("Tokenizing dataset...")
    tokenized_train = dataset["train"].map(tokenize, batched=True)
    tokenized_test = dataset["test"].map(tokenize, batched=True)

    # Extract fields for encodings and labels
    train_encodings = {key: tokenized_train[key] for key in ["input_ids", "attention_mask", "token_type_ids"]}
    test_encodings = {key: tokenized_test[key] for key in ["input_ids", "attention_mask", "token_type_ids"]}
    train_labels = tokenized_train["label"]
    test_labels = tokenized_test["label"]

    return (
        TextDataset(train_encodings, train_labels),
        TextDataset(test_encodings, test_labels),
        len(set(train_labels)),  # Number of unique labels
    )

# Step 3: Training Function
def train_model(train_loader, model, optimizer, device, num_epochs=3):
    """
    Train the BERT model using PyTorch.
    :param train_loader: DataLoader for the training dataset.
    :param model: Pre-trained BERT model for sequence classification.
    :param optimizer: Optimizer for training.
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param num_epochs: Number of epochs to train.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, labels in train_loader:
            # Move data to the specified device
            batch = {key: val.to(device) for key, val in batch.items()}
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            # Backward pass
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Step 4: Evaluation Function
def evaluate_model(test_loader, model, device):
    """
    Evaluate the model's accuracy on the test dataset.
    :param test_loader: DataLoader for the test dataset.
    :param model: Fine-tuned BERT model.
    :param device: Device to run the evaluation on ('cpu' or 'cuda').
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch, batch_labels in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(labels, preds)
    print(f"Test Accuracy: {accuracy:.4f}")

# Step 5: Main Function
if __name__ == "__main__":
    # Configuration
    dataset_name = "ag_news"  # Dataset (e.g., 'ag_news', 'imdb')
    model_name = "bert-base-uncased"  # Pre-trained model
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    print("Preprocessing data...")
    train_dataset, test_dataset, num_labels = preprocess_data(dataset_name, model_name)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load pre-trained BERT model
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    print("Training model...")
    train_model(train_loader, model, optimizer, device, num_epochs)

    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(test_loader, model, device)
