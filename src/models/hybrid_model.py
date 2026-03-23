"""
Advanced Hybrid Deep Learning model for predictive alerting.
Uses a CNN to extract local trends, an LSTM to evaluate sequence causation,
and a skip-connection to the raw latest metrics for precise thresholding.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src import config


class HybridNet(nn.Module):
    def __init__(self, input_dim=config.NUM_FEATURES, hidden_dim=64, num_layers=2, dropout=0.3):
        super(HybridNet, self).__init__()

        # 1. Trend Extractor (CNN)
        # 15-minute kernel smooths noise and identifies slopes
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=15, padding=7)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 2. Sequential Memory (LSTM)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Decision Engine with Skip Connection
        # We add 'input_dim' to the dense layer to concatenate the raw final timestep
        self.fc1 = nn.Linear(hidden_dim + input_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Save the raw final timestep for the skip connection
        raw_latest_metrics = x[:, -1, :]

        # CNN Pass
        x_cnn = x.transpose(1, 2)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)

        # LSTM Pass
        lstm_out, _ = self.lstm(x_cnn)
        lstm_latest_state = lstm_out[:, -1, :]

        # skip connection: Combine LSTM memory with the exact current raw state
        combined = torch.cat((lstm_latest_state, raw_latest_metrics), dim=1)

        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits


class HybridAlertingModel:
    def __init__(self, hidden_dim=64, num_layers=2, epochs=40, batch_size=256, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Hybrid CNN-LSTM on device: {self.device}")

        self.model = HybridNet(
            input_dim=config.NUM_FEATURES,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        self.scaler = StandardScaler()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.is_trained = False

    def _scale_3d_array(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        samples, timesteps, features = x.shape
        x_flat = x.reshape(-1, features)
        if fit:
            x_scaled_flat = self.scaler.fit_transform(x_flat)
        else:
            x_scaled_flat = self.scaler.transform(x_flat)
        return x_scaled_flat.reshape(samples, timesteps, features)

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        print(f"Scaling 3D training data and moving to {self.device}...")
        x_train_scaled = self._scale_3d_array(x_train, fit=True)

        val_split = int(len(x_train_scaled) * 0.9)
        x_train_split, x_val_split = x_train_scaled[:val_split], x_train_scaled[val_split:]
        y_train_split, y_val_split = y_train[:val_split], y_train[val_split:]

        train_dataset = TensorDataset(torch.tensor(x_train_split, dtype=torch.float32),
                                      torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(x_val_split, dtype=torch.float32),
                                    torch.tensor(y_val_split, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Balanced weight cap for clean data
        num_positives = np.sum(y_train_split)
        num_negatives = len(y_train_split) - num_positives
        tamed_weight = min(10.0, num_negatives / (num_positives + 1e-5))

        pos_weight = torch.tensor([tamed_weight], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        print(f"Starting Training (Max {self.epochs} Epochs, Patience: 6)...")
        best_val_loss = float('inf')
        patience, patience_counter = 6, 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    val_loss += criterion(self.model(batch_x), batch_y).item()

            avg_train, avg_val = train_loss / len(train_loader), val_loss / len(val_loader)
            scheduler.step(avg_val)

            print(f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if avg_val < best_val_loss:
                best_val_loss, patience_counter = avg_val, 0
                import copy
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}! Restoring best weights.")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_trained = True
        print("Hybrid Model Training complete.")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")

        x_scaled = self._scale_3d_array(x, fit=False)
        self.model.eval()

        all_probs = []
        with torch.no_grad():
            dataset = TensorDataset(torch.tensor(x_scaled, dtype=torch.float32))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for batch_x in dataloader:
                probs = torch.sigmoid(self.model(batch_x[0].to(self.device))).cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs).flatten()

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'scaler': self.scaler}, filepath)
        print(f"Hybrid Model and Scaler saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved artifact found at {filepath}")
        artifact = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(artifact['model_state_dict'])
        self.scaler = artifact['scaler']
        self.is_trained = True
        print(f"Hybrid Model loaded from {filepath}")
