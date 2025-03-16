"""
This module provides functions for training and evaluating deep learning models
for classification tasks. It includes utilities for:
    - splitting data into training, validation, and test sets
    - normalizing data
    - training a deep learning model with early stopping and learning rate scheduling
    - visualizing training progress
    - evaluating a model using various metrics
    - generating a classification report based on model performance
The main function, `modelling_main`, orchestrates the data split, model training,
and evaluation process.
Supported model architectures (by data dimensionality):
    - 2D: LeNet2D, AlexNet2D
    - 3D: LeNet3D, AlexNet3D
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from _9_models_architecture import LeNet2D, LeNet3D, AlexNet2D, AlexNet3D
from collections import Counter


def find_best_split(X_train, y_train, start=0.19, end=0.21, steps=100, min_val_size=10):
    """
    Find the best validation split size based on the class distribution difference.

    Parameters:
        X_train (numpy.ndarray): training data
        y_train (numpy.ndarray): training labels
        start (float, optional): starting fraction for validation set size (default: 0.19)
        end (float, optional): ending fraction for validation set size (default: 0.21)
        steps (int, optional): number of steps between start and end (default: 100)
        min_val_size (int, optional): minimum number of samples required in the validation set (default: 10)

    Returns:
        float: the best validation split size
    """
    def calculate_class_distribution(y):
        counter = Counter(y)
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}

    def calculate_distribution_difference(y_train, y_val):
        dist_train = calculate_class_distribution(y_train)
        dist_val = calculate_class_distribution(y_val)
        classes = set(dist_train.keys()).union(set(dist_val.keys()))
        diff = sum(abs(dist_train.get(cls, 0) - dist_val.get(cls, 0)) for cls in classes)
        return diff

    best_split = None
    best_diff = float('inf')

    for val_size in np.linspace(start, end, steps):
        _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)
        if len(y_val) < min_val_size:
            continue
        diff = calculate_distribution_difference(y_train, y_val)
        if diff < best_diff:
            best_diff = diff
            best_split = val_size

    return best_split


def split_data(X_train, y_train, min_val_size=10):
    """
    Split the data into training and validation sets using a stratified approach.

    Parameters:
        X_train (numpy.ndarray): training data
        y_train (numpy.ndarray): training labels
        min_val_size (int, optional): minimum number of samples required in the validation set (default: 10)

    Returns:
        tuple:
            - X_train (numpy.ndarray): updated training data
            - X_val (numpy.ndarray): validation data
            - y_train (numpy.ndarray): updated training labels
            - y_val (numpy.ndarray): validation labels
    """
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=find_best_split(X_train, y_train, min_val_size=min_val_size),
            stratify=y_train,
            random_state=42
        )
    except ValueError as e:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=find_best_split(X_train, y_train, min_val_size=min_val_size),
            stratify=None,
            random_state=42
        )
    return X_train, X_val, y_train, y_val


def normalize_data(X_train, X_val, X_test):
    """
    Normalize the training, validation, and test data using the mean and standard deviation of X_train.

    Parameters:
        X_train (numpy.ndarray): training data
        X_val (numpy.ndarray): validation data
        X_test (numpy.ndarray): test data

    Returns:
        tuple:
            - X_train (numpy.ndarray): normalized training data
            - X_val (numpy.ndarray): normalized validation data
            - X_test (numpy.ndarray): normalized test data
    """
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_val, X_test


def train_model(model, train_loader, val_loader, num_epochs, patience, learning_rate, device, weight_decay, dropout_prob):
    """
    Train a deep learning model with early stopping and learning rate scheduling.

    Parameters:
        model (torch.nn.Module): the model to be trained.
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int): maximum number of epochs
        patience (int): patience for early stopping
        learning_rate (float): initial learning rate
        device (torch.device): device to run training on
        weight_decay (float): weight decay for optimizer
        dropout_prob (float): wropout probability for the model

    Returns:
        tuple:
            - train_losses (list): training losses per epoch
            - val_losses (list): validation losses per epoch
            - train_accuracies (list): training accuracies per epoch
            - val_accuracies (list): validation accuracies per epoch
            - f1_scores_train (list): training F1 scores per epoch
            - f1_scores_val (list): validation F1 scores per epoch
            - model (torch.nn.Module): best model instance after training
            - epoch_counter (int): number of epochs trained
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    epoch_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    f1_scores_train, f1_scores_val = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        f1_train = f1_score(all_labels, all_preds, average='weighted')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        f1_scores_train.append(f1_train)

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader)
        val_accuracy = correct / total
        f1_val = f1_score(all_labels, all_preds, average='weighted')

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        f1_scores_val.append(f1_val)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < learning_rate:
            learning_rate = current_lr

        if val_loss < best_loss or val_accuracy > best_val_accuracy:
            best_loss = min(best_loss, val_loss)
            best_val_accuracy = max(best_val_accuracy, val_accuracy)
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return (train_losses, val_losses, train_accuracies, val_accuracies,
            f1_scores_train, f1_scores_val, model, epoch_counter)


def visualize_results(train_losses, val_losses, train_accuracies, val_accuracies, f1_train, f1_val):
    """
    Visualize training progress including loss, accuracy, and F1 score over epochs.

    Parameters:
        train_losses (list): training losses per epoch
        val_losses (list): validation losses per epoch
        train_accuracies (list): training accuracies per epoch
        val_accuracies (list): validation accuracies per epoch
        f1_train (list): training F1 scores per epoch
        f1_val (list): validation F1 scores per epoch

    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_train, label='Training F1 Score')
    plt.plot(epochs, f1_val, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score Over Epochs')

    plt.show()


def evaluate_model(model, data_loader, num_classes):
    """
    Evaluate a trained model on a given dataset and compute performance metrics.

    Parameters:
        model (torch.nn.Module): trained model
        data_loader (DataLoader): dataLoader for the evaluation dataset
        num_classes (int): number of classes

    Returns:
        dict: dictionary containing accuracy, precision, recall, F1 score, and log loss
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logloss = log_loss(all_labels, np.array(all_probs), labels=np.arange(num_classes))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'log_loss': logloss
    }
    return metrics


def generate_classification_report(model, num_classes, train_loader, val_loader, test_loader):
    """
    Generate a classification report by evaluating the model on training, validation, and test sets.

    Parameters:
        model (torch.nn.Module): trained model
        num_classes (int): number of classes
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        test_loader (DataLoader): DataLoader for test data

    Returns:
        dict: dictionary with keys 'train', 'validation', and 'test', where each value is a dictionary of evaluation metrics
    """
    metrics = {
        'train': evaluate_model(model, train_loader, num_classes),
        'validation': evaluate_model(model, val_loader, num_classes),
        'test': evaluate_model(model, test_loader, num_classes)
    }
    return metrics


def modelling_main(X_train, y_train, X_test, y_test, alpha, nn_model='LeNet', num_epochs=100, patience=15,
                   batch_size=32, learning_rate=0.0005, weight_decay=0.01, dropout_prob=0.35,
                   device=torch.device("cpu")):
    """
    Train and evaluate a deep learning model for classification.

    This function splits the training data into training and validation sets,
    selects the appropriate model architecture based on the data shape, trains the model
    with early stopping and learning rate scheduling, and generates performance metrics.

    Supported model architectures:
        - 2D: LeNet2D, AlexNet2D
        - 3D: LeNet3D, AlexNet3D

    Parameters:
        X_train (numpy.ndarray): training data
        y_train (numpy.ndarray): training labels
        X_test (numpy.ndarray): test data
        y_test (numpy.ndarray): test labels
        alpha (int): parameter for the model architecture
        nn_model (str, optional): model name to use ('LeNet' or 'AlexNet')
        num_epochs (int, optional): maximum number of training epochs
        patience (int, optional): early stopping patience
        batch_size (int, optional): batch size for training
        learning_rate (float, optional): initial learning rate
        weight_decay (float, optional): weight decay for optimizer
        dropout_prob (float, optional): dropout probability
        device (torch.device, optional): device for training

    Returns:
        tuple:
            - metrics_generated (dict): dictionary containing evaluation metrics for train, validation, and test sets
            - epoch_counter (int): number of epochs completed during training
    """
    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    model_dict_2d = {
        'LeNet': LeNet2D,
        'AlexNet': AlexNet2D
    }

    model_dict_3d = {
        'LeNet': LeNet3D,
        'AlexNet': AlexNet3D
    }

    num_classes = len(np.unique(y_train))

    if len(X_train.shape) == 3:
        # For 2D data, add a channel dimension.
        X_train = X_train[:, np.newaxis, :, :]
        X_val = X_val[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]
        model_fn = model_dict_2d.get(nn_model)
        model = model_fn(num_classes=num_classes, alpha=alpha, dropout_rate=dropout_prob).to(device)
    elif len(X_train.shape) == 4:
        # For 3D data, add a channel dimension.
        X_train = X_train[:, np.newaxis, :, :, :]
        X_val = X_val[:, np.newaxis, :, :, :]
        X_test = X_test[:, np.newaxis, :, :, :]
        model_fn = model_dict_3d.get(nn_model)
        model = model_fn(num_classes=num_classes, alpha=alpha, dropout_rate=dropout_prob).to(device)
    else:
        raise ValueError("Unsupported data shape for X_train")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    (train_losses, val_losses, train_accuracies, val_accuracies,
     f1_train, f1_val, model_instance, epoch_counter) = train_model(
        model, train_loader, val_loader, num_epochs, patience,
        learning_rate, device, weight_decay, dropout_prob
    )

    metrics_generated = generate_classification_report(model_instance, num_classes,
                                                         train_loader, val_loader, test_loader)

    return metrics_generated, epoch_counter
