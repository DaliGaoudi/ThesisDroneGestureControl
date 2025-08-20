import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class GestureModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_training = False
        self.training_data = []
        self.training_labels = []
        
    def start_training(self):
        """Start collecting training data."""
        self.is_training = True
        self.training_data = []
        self.training_labels = []
        
    def stop_training(self):
        """Stop collecting training data and train the model."""
        self.is_training = False
        if len(self.training_data) > 0:
            self._train_model()
            
    def add_training_sample(self, points, label):
        """
        Add a training sample.
        
        Args:
            points: List of (x, y) coordinates of the finger movement
            label: 'clockwise', 'counterclockwise', or 'none'
        """
        if not self.is_training:
            return
            
        # Convert points to features
        features = self._extract_features(points)
        self.training_data.append(features)
        self.training_labels.append(label)
        
    def _extract_features(self, points):
        """
        Extract features from a sequence of points.
        
        Args:
            points: List of (x, y) coordinates
            
        Returns:
            numpy array of features
        """
        if len(points) < 2:
            return np.zeros(10)  # Return zeros if not enough points
            
        points = np.array(points)
        
        # Calculate basic features
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        
        # Calculate angles between consecutive movements
        angles = np.arctan2(dy, dx)
        angle_changes = np.diff(angles)
        
        # Calculate total distance
        distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate features
        features = np.array([
            np.mean(angle_changes),  # Average angle change
            np.std(angle_changes),   # Standard deviation of angle changes
            np.sum(angle_changes),   # Total angle change
            np.mean(distances),      # Average distance between points
            np.std(distances),       # Standard deviation of distances
            np.max(distances),       # Maximum distance
            np.min(distances),       # Minimum distance
            np.sum(distances),       # Total distance
            np.max(points[:, 0]) - np.min(points[:, 0]),  # Width of movement
            np.max(points[:, 1]) - np.min(points[:, 1])   # Height of movement
        ])
        
        return features
        
    def _train_model(self):
        """Train the model on collected data."""
        if len(self.training_data) < 2:
            return
            
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Scale the features
        X = self.scaler.fit_transform(X)
        
        # Train a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
    def predict(self, points):
        """
        Predict the gesture from a sequence of points.
        
        Args:
            points: List of (x, y) coordinates of the finger movement
            
        Returns:
            'clockwise', 'counterclockwise', or 'none'
        """
        if self.model is None:
            return 'none'
            
        features = self._extract_features(points)
        features = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(features)[0]
        
    def save_model(self, filename='gesture_model.joblib'):
        """Save the trained model to a file."""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, filename)
            
    def load_model(self, filename='gesture_model.joblib'):
        """Load a trained model from a file."""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler'] 