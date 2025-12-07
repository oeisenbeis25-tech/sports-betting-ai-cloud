"""
Logistic Regression Model - Built from scratch
Implements a binary classification model for sports prediction
"""

import numpy as np


class SportsPredictor:
    """
    Logistic Regression classifier for sports betting predictions
    Built entirely from scratch without using sklearn
    """
    
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Initialize the model
        
        Args:
            learning_rate: Learning rate for gradient descent
            epochs: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Average loss
        """
        # Avoid log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def train(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (binary: 0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        print(f"    Starting training with {n_samples} samples...")
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass: compute predictions
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass: compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")
        
        print(f"    Training complete! Final loss: {self.losses[-1]:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probability of each class
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Probabilities for class 0 and class 1
        """
        z = np.dot(X, self.weights) + self.bias
        proba_class_1 = self.sigmoid(z)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack((proba_class_0, proba_class_1))
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold
            
        Returns:
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate model accuracy on given data
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score (0-1)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_model_info(self):
        """Get model information"""
        return {
            'weights_shape': self.weights.shape if self.weights is not None else None,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'total_loss_history': len(self.losses),
            'final_loss': self.losses[-1] if self.losses else None
        }
