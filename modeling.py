import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

# Generate random time-series data and corresponding labels
data = np.random.randn(1000, 10)  # 1000 windows of length 10
labels = np.random.randint(0, 2, size=(1000,))  # binary labels for each window

# Split data into training and validation sets
split = int(len(data) * 0.8)  # use 80% of the data for training
train_data, train_labels = data[:split], labels[:split]
val_data, val_labels = data[split:], labels[split:]

# Convert data into PyTorch tensors
train_data, train_labels = torch.FloatTensor(train_data), torch.LongTensor(train_labels)
val_data, val_labels = torch.FloatTensor(val_data), torch.LongTensor(val_labels)

# Convert labels into one-hot encoded format
num_classes = 2  # binary classification problem
train_labels = nn.functional.one_hot(train_labels, num_classes=num_classes)
val_labels = nn.functional.one_hot(val_labels, num_classes=num_classes)

# Create PyTorch datasets and data loaders
batch_size = 64
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define hyperparameters
input_size = 10  # number of features in each window
hidden_size = 20  # number of neurons in hidden layer
output_size = 2  # number of classes
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Define the model, loss function, and optimizer
model = Net(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define the training and validation datasets and loaders
train_data = np.random.randn(1000, 10)  # 1000 training windows of length 10
train_labels = np.random.randint(0, 2, size=(1000,))  # binary labels for each training window
train_data, train_labels = torch.FloatTensor(train_data), torch.LongTensor(train_labels)
train_labels = nn.functional.one_hot(train_labels, num_classes=output_size)

val_data = np.random.randn(200, 10)  # 200 validation windows of length 10
val_labels = np.random.randint(0, 2, size=(200,))  # binary labels for each validation window
val_data, val_labels = torch.FloatTensor(val_data), torch.LongTensor(val_labels)
val_labels = nn.functional.one_hot(val_labels, num_classes=output_size)

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Train the model
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_train_acc = 0.0

    model.train()
    for data, label in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label.argmax(dim=1))
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * data.size(0)
        epoch_train_acc += (output.argmax(dim=1) == label.argmax(dim=1)).sum().item()

    # Evaluate on validation set
    epoch_val_loss = 0.0
    epoch_val_acc = 0.0

    model.eval()
    with torch.no_grad():
        for data, label in val_loader:
            output = model(data)
            loss = criterion(output, label.argmax(dim=1))

            epoch_val_loss += loss.item() * data.size(0)
            epoch_val_acc += (output.argmax(dim=1) == label.argmax(dim=1)).sum().item()

    # Print statistics for current epoch
    epoch_train_loss /= len(train_dataset)
    epoch_train_acc /= len(train_dataset)
    epoch_val_loss /= len(val_dataset)
    epoch_val_acc /= len(val_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    val_loss.append(val_loss)
    val_acc.append(val_acc)

import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_acc(train_acc, val_acc):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    classes = ['No Spike', 'Spike']
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_confusion_matrix(predictions, labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    return cm

# Plot the results
#plot_loss(train_loss, val_loss)
#plot_acc(train_acc, val_acc)
#plot_confusion_matrix(confusion_matrix)