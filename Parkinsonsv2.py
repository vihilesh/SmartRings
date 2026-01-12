"""
Complete Parkinson's Detection System with Machine Learning
Combines feature extraction with ML classification

USAGE:
1. Extract features from accelerometer data (statistical/signal processing)
2. Train ML models on labeled patient data
3. Predict Parkinson's disease for new patients

IMPORTANT: Requires real patient data with ground truth diagnoses for training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress, spearmanr
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_curve, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib


# ============================================================================
# PART 1: FEATURE EXTRACTION (Statistical/Signal Processing)
# ============================================================================

class MicrographiaDetector:
    """Detects progressive micrographia from accelerometer data"""
    
    def __init__(self, sampling_rate: int = 50, segment_duration: float = 2.0):
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sampling_rate)
    
    def preprocess_signal(self, acc_data: np.ndarray) -> np.ndarray:
        """Preprocess accelerometer data"""
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        acc_magnitude = acc_magnitude - np.mean(acc_magnitude)
        
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 20.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        acc_filtered = signal.filtfilt(b, a, acc_magnitude)
        
        return np.abs(acc_filtered)
    
    def segment_signal(self, acc_magnitude: np.ndarray) -> List[np.ndarray]:
        """Split signal into temporal segments"""
        num_segments = len(acc_magnitude) // self.segment_samples
        segments = []
        
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segments.append(acc_magnitude[start:end])
        
        return segments
    
    def extract_amplitude_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Extract amplitude-related features from a segment"""
        features = {}
        
        features['mean_amplitude'] = np.mean(segment)
        features['median_amplitude'] = np.median(segment)
        features['max_amplitude'] = np.max(segment)
        features['std_amplitude'] = np.std(segment)
        features['amplitude_range'] = np.ptp(segment)
        features['p75_amplitude'] = np.percentile(segment, 75)
        features['p90_amplitude'] = np.percentile(segment, 90)
        features['p95_amplitude'] = np.percentile(segment, 95)
        
        peaks, properties = signal.find_peaks(segment, height=np.mean(segment))
        if len(peaks) > 0:
            features['mean_peak_height'] = np.mean(properties['peak_heights'])
            features['num_peaks'] = len(peaks)
        else:
            features['mean_peak_height'] = 0
            features['num_peaks'] = 0
        
        features['signal_energy'] = np.sum(segment ** 2)
        features['rms_amplitude'] = np.sqrt(np.mean(segment ** 2))
        
        return features
    
    def detect_progressive_reduction(
        self, 
        segment_features: List[Dict[str, float]], 
        feature_key: str = 'mean_amplitude'
    ) -> Dict[str, float]:
        """Analyze temporal trend in amplitude features"""
        values = np.array([f[feature_key] for f in segment_features])
        segment_indices = np.arange(len(values))
        
        slope, intercept, r_value, p_value, std_err = linregress(
            segment_indices, values
        )
        
        spearman_corr, spearman_p = spearmanr(segment_indices, values)
        
        if values[0] != 0:
            pct_change = 100 * (values[-1] - values[0]) / values[0]
        else:
            pct_change = 0
        
        first_third = values[:len(values)//3]
        last_third = values[-len(values)//3:]
        early_mean = np.mean(first_third)
        late_mean = np.mean(last_third)
        
        if early_mean != 0:
            early_late_pct_change = 100 * (late_mean - early_mean) / early_mean
        else:
            early_late_pct_change = 0
        
        pooled_std = np.sqrt((np.std(first_third)**2 + np.std(last_third)**2) / 2)
        if pooled_std > 0:
            cohens_d = (early_mean - late_mean) / pooled_std
        else:
            cohens_d = 0
        
        return {
            'slope': slope,
            'slope_normalized': slope / (np.mean(values) + 1e-6),
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'pct_change': pct_change,
            'early_late_pct_change': early_late_pct_change,
            'cohens_d': cohens_d
        }
    
    def analyze_sample(self, acc_data: np.ndarray) -> Dict[str, any]:
        """Complete analysis pipeline for one handwriting sample"""
        acc_magnitude = self.preprocess_signal(acc_data)
        segments = self.segment_signal(acc_magnitude)
        
        if len(segments) < 3:
            return {'error': 'Insufficient data - need at least 3 segments'}
        
        segment_features = [
            self.extract_amplitude_features(seg) for seg in segments
        ]
        
        results = {}
        important_features = [
            'mean_amplitude', 'p90_amplitude', 'rms_amplitude', 
            'max_amplitude', 'signal_energy'
        ]
        
        for feature in important_features:
            trend = self.detect_progressive_reduction(segment_features, feature)
            results[feature] = trend
        
        return results


class ParkinsonsFeaturesExtractor:
    """Comprehensive feature extraction for Parkinson's detection"""
    
    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate
        self.micrographia_detector = MicrographiaDetector(sampling_rate)
    
    def extract_all_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive feature set for Parkinson's"""
        features = {}
        
        # 1. Micrographia features
        micro_results = self.micrographia_detector.analyze_sample(acc_data)
        if 'error' not in micro_results:
            for feat_name, trend in micro_results.items():
                features[f'micro_{feat_name}_slope'] = trend['slope']
                features[f'micro_{feat_name}_pct_change'] = trend['early_late_pct_change']
                features[f'micro_{feat_name}_r_squared'] = trend['r_squared']
        
        # 2. Tremor features (4-6 Hz)
        tremor_features = self._extract_tremor_features(acc_data)
        features.update(tremor_features)
        
        # 3. Velocity features
        velocity_features = self._extract_velocity_features(acc_data)
        features.update(velocity_features)
        
        # 4. Smoothness (jerk)
        jerk_features = self._extract_jerk_features(acc_data)
        features.update(jerk_features)
        
        # 5. Rhythm/regularity
        rhythm_features = self._extract_rhythm_features(acc_data)
        features.update(rhythm_features)
        
        # 6. Overall amplitude statistics
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        features['overall_mean_amplitude'] = np.mean(acc_magnitude)
        features['overall_std_amplitude'] = np.std(acc_magnitude)
        features['overall_cv_amplitude'] = np.std(acc_magnitude) / (np.mean(acc_magnitude) + 1e-6)
        
        return features
    
    def _extract_tremor_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract tremor-specific features (Parkinson's: 4-6 Hz)"""
        from scipy.signal import welch
        
        features = {}
        
        for axis, axis_name in enumerate(['x', 'y', 'z']):
            freqs, psd = welch(acc_data[:, axis], fs=self.sampling_rate, nperseg=256)
            
            tremor_band = (freqs >= 4) & (freqs <= 6)
            tremor_power = np.sum(psd[tremor_band])
            total_power = np.sum(psd)
            
            features[f'{axis_name}_tremor_power'] = tremor_power
            features[f'{axis_name}_tremor_ratio'] = tremor_power / (total_power + 1e-6)
            
            if np.any(tremor_band):
                tremor_freqs = freqs[tremor_band]
                tremor_psd = psd[tremor_band]
                features[f'{axis_name}_tremor_peak_freq'] = tremor_freqs[np.argmax(tremor_psd)]
            else:
                features[f'{axis_name}_tremor_peak_freq'] = 0
        
        return features
    
    def _extract_velocity_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract velocity-related features"""
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        
        velocity = np.cumsum(acc_magnitude) / self.sampling_rate
        velocity = velocity - np.mean(velocity)
        
        return {
            'mean_velocity': np.mean(np.abs(velocity)),
            'std_velocity': np.std(velocity),
            'velocity_range': np.ptp(velocity),
            'velocity_cv': np.std(velocity) / (np.mean(np.abs(velocity)) + 1e-6)
        }
    
    def _extract_jerk_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract jerk (smoothness) features"""
        dt = 1.0 / self.sampling_rate
        jerk = np.diff(acc_data, axis=0) / dt
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        
        duration = len(acc_data) * dt
        normalized_jerk = np.sqrt(np.sum(jerk_magnitude**2) * dt**5) / (duration**3)
        
        return {
            'normalized_jerk': normalized_jerk,
            'mean_jerk': np.mean(jerk_magnitude),
            'std_jerk': np.std(jerk_magnitude),
            'max_jerk': np.max(jerk_magnitude)
        }
    
    def _extract_rhythm_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract rhythm/regularity features"""
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acc_magnitude, distance=self.sampling_rate//10)
        
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sampling_rate
            
            return {
                'mean_stroke_interval': np.mean(intervals),
                'std_stroke_interval': np.std(intervals),
                'cv_stroke_interval': np.std(intervals) / (np.mean(intervals) + 1e-6),
                'num_strokes': len(peaks)
            }
        else:
            return {
                'mean_stroke_interval': 0,
                'std_stroke_interval': 0,
                'cv_stroke_interval': 0,
                'num_strokes': len(peaks)
            }


# ============================================================================
# PART 2: MACHINE LEARNING PIPELINE
# ============================================================================

class ParkinsonsMLClassifier:
    """
    Machine Learning pipeline for Parkinson's disease classification
    """
    
    def __init__(self, feature_extractor: ParkinsonsFeaturesExtractor):
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.feature_names = None
        self.selected_feature_indices = None
        
    def prepare_dataset(self, patient_data: List[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from patient accelerometer data
        
        Args:
            patient_data: List of (acc_data, label) tuples
                         label: 0=healthy, 1=Parkinson's
        
        Returns:
            X: Feature matrix (n_patients, n_features)
            y: Labels (n_patients,)
        """
        features_list = []
        labels = []
        
        print("Extracting features from patient data...")
        for i, (acc_data, label) in enumerate(patient_data):
            print(f"  Processing patient {i+1}/{len(patient_data)}...")
            features = self.feature_extractor.extract_all_features(acc_data)
            features_list.append(features)
            labels.append(label)
        
        # Convert to DataFrame for easier handling
        features_df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        # Convert to numpy arrays
        X = features_df.values
        y = np.array(labels)
        
        print(f"\nDataset prepared: {X.shape[0]} patients, {X.shape[1]} features")
        print(f"  Healthy controls: {np.sum(y==0)}")
        print(f"  Parkinson's patients: {np.sum(y==1)}")
        
        return X, y
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, k: int = 10, method: str = 'anova'):
        """
        Select most important features
        
        Args:
            X: Feature matrix
            y: Labels
            k: Number of features to select
            method: 'anova' or 'rfe'
        """
        print(f"\nPerforming feature selection ({method}, k={k})...")
        
        if method == 'anova':
            # Univariate feature selection
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_feature_indices = self.feature_selector.get_support(indices=True)
            
        elif method == 'rfe':
            # Recursive feature elimination
            estimator = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
            self.feature_selector = RFE(estimator, n_features_to_select=k, step=1)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_feature_indices = np.where(self.feature_selector.support_)[0]
        
        selected_names = [self.feature_names[i] for i in self.selected_feature_indices]
        print(f"Selected features: {selected_names}")
        
        return X_selected
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'logistic'):
        """
        Train ML model
        
        Args:
            X: Feature matrix
            y: Labels
            model_type: 'logistic', 'rf', 'svm', 'gb'
        """
        print(f"\nTraining {model_type} model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model based on type
        if model_type == 'logistic':
            self.model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=0.1,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model on all data
        self.model.fit(X_scaled, y)
        
        print(f"Model trained successfully")
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_method: str = 'loocv') -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv_method: 'loocv' (Leave-One-Out) or integer for k-fold
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nPerforming cross-validation ({cv_method})...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Set up cross-validation
        if cv_method == 'loocv':
            cv = LeaveOneOut()
        else:
            cv = int(cv_method)
        
        # Get predictions
        y_pred = cross_val_predict(self.model, X_scaled, y, cv=cv)
        y_pred_proba = cross_val_predict(self.model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  AUC: {auc:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Healthy', 'Parkinson\'s']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Healthy  PD")
        print(f"Actual Healthy   {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"       PD        {cm[1,0]:3d}   {cm[1,1]:3d}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'y_true': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def predict(self, acc_data: np.ndarray) -> Dict:
        """
        Predict Parkinson's disease for new patient
        
        Args:
            acc_data: Accelerometer data (N, 3)
        
        Returns:
            Dictionary with prediction, probability, and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(acc_data)
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            feature_vector = feature_vector[self.selected_feature_indices]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Determine confidence
        max_prob = np.max(probability)
        if max_prob > 0.8:
            confidence = 'high'
        elif max_prob > 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'prediction': 'Parkinson\'s' if prediction == 1 else 'Healthy',
            'probability_parkinsons': probability[1],
            'probability_healthy': probability[0],
            'confidence': confidence
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if hasattr(self.model, 'coef_'):
            # Logistic regression or SVM
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")
        
        # Get feature names (accounting for feature selection)
        if self.feature_selector is not None:
            feature_names = [self.feature_names[i] for i in self.selected_feature_indices]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_feature_indices': self.selected_feature_indices
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.selected_feature_indices = model_data['selected_feature_indices']
        print(f"Model loaded from {filepath}")
    
    def plot_roc_curve(self, cv_results: Dict):
        """Plot ROC curve from cross-validation results"""
        fpr, tpr, thresholds = roc_curve(cv_results['y_true'], cv_results['y_pred_proba'])
        auc = cv_results['auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Parkinson\'s Detection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 3: UTILITY FUNCTIONS
# ============================================================================

def generate_simulated_data(duration=20, sampling_rate=50, has_parkinsons=False):
    """
    Generate synthetic accelerometer data for testing
    
    WARNING: Only for algorithm testing, NOT for training ML models!
    """
    t = np.linspace(0, duration, duration * sampling_rate)
    base_freq = 2.0
    
    if has_parkinsons:
        # Micrographia: amplitude decreases
        amplitude = np.linspace(1.0, 0.3, len(t))
        
        # Add tremor at 4-6 Hz
        tremor_freq = 5.0
        tremor = 0.15 * np.sin(2 * np.pi * tremor_freq * t)
        
        # Higher jerk (less smooth)
        jerk_factor = 1.5
    else:
        # Normal: constant amplitude
        amplitude = np.ones(len(t))
        tremor = 0
        jerk_factor = 1.0
    
    # Simulate writing strokes
    signal_x = amplitude * np.sin(2 * np.pi * base_freq * t) + tremor
    signal_y = amplitude * np.cos(2 * np.pi * base_freq * 1.5 * t) + tremor
    signal_z = amplitude * 0.5 * np.sin(2 * np.pi * base_freq * 0.8 * t)
    
    # Add noise (more for Parkinson's)
    noise_level = 0.15 if has_parkinsons else 0.1
    noise = np.random.normal(0, noise_level, (len(t), 3))
    
    # Add jerk
    if has_parkinsons:
        jerk_noise = np.random.normal(0, 0.2, (len(t), 3))
        noise = noise + jerk_noise
    
    acc_data = np.column_stack([signal_x, signal_y, signal_z]) + noise
    
    return acc_data


def load_colmi_data(filepath):
    """Load accelerometer data from Colmi ring - adjust to your format"""
    df = pd.read_csv(filepath)
    acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
    return acc_data

def load_colmi_data2(filepath):
    """Load accelerometer data from Colmi ring - adjust to your format"""
    df = pd.read_csv(filepath)
    acc_data = df[['accX', 'accY', 'accZ']].values
    return acc_data


# ============================================================================
# PART 4: COMPLETE EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPLETE PARKINSON'S DETECTION SYSTEM WITH MACHINE LEARNING")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Generate Simulated Dataset (replace with real data!)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Preparing Dataset")
    print("="*80)
    print("\nWARNING: Using simulated data for demonstration.")
    print("Replace with real patient data for actual use!\n")
    
    # Simulate 30 patients (15 healthy, 15 Parkinson's)
    patient_data = []
    
    print("Generating simulated patient data...")
    for i in range(15):
        acc_data = generate_simulated_data(duration=20, has_parkinsons=False)
        patient_data.append((acc_data, 0))  # 0 = healthy

    print("Generating simulated patient data...")
    
    for i in range(15):
        acc_data = generate_simulated_data(duration=20, has_parkinsons=True)
        patient_data.append((acc_data, 1))  # 1 = Parkinson's
    
    # print(f"Generated {len(patient_data)} simulated patients") """ """

    
    # print("loading colmi patient data...")

    # acc_data = load_colmi_data2("/data/POO5SentTrial2.ring_data_20251221_234508.csv")
    # patient_data.append((acc_data, 1))  # 1 = Parkinson's

    
    # ========================================================================
    # STEP 2: Extract Features
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Feature Extraction")
    print("="*80)
    
    feature_extractor = ParkinsonsFeaturesExtractor(sampling_rate=50)
    ml_classifier = ParkinsonsMLClassifier(feature_extractor)
    
    X, y = ml_classifier.prepare_dataset(patient_data)
    
    # ========================================================================
    # STEP 3: Feature Selection
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Feature Selection")
    print("="*80)
    
    # Select top 10 features
    X_selected = ml_classifier.feature_selection(X, y, k=10, method='anova')
    
    # ========================================================================
    # STEP 4: Train Multiple Models and Compare
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Model Training and Evaluation")
    print("="*80)
    
    model_types = ['logistic', 'rf', 'svm', 'gb']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"Training and evaluating: {model_type.upper()}")
        print(f"{'='*80}")
        
        # Train model
        ml_classifier.train_model(X_selected, y, model_type=model_type)
        
        # Cross-validate
        cv_results = ml_classifier.cross_validate(X_selected, y, cv_method='loocv')
        results[model_type] = cv_results
        
        # Feature importance (if supported)
        try:
            importance = ml_classifier.get_feature_importance()
            print(f"\nTop 5 Most Important Features:")
            print(importance.head(5).to_string(index=False))
        except:
            pass
    
    # ========================================================================
    # STEP 5: Compare Models
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Model Comparison")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': model_types,
        'Accuracy': [results[m]['accuracy'] for m in model_types],
        'AUC': [results[m]['auc'] for m in model_types],
        'F1 Score': [results[m]['f1'] for m in model_types]
    }).sort_values('AUC', ascending=False)
    
    print("\nModel Performance Comparison:")
    print(comparison.to_string(index=False))
    
    best_model = comparison.iloc[0]['Model']
    print(f"\nBest performing model: {best_model.upper()}")
    
    # ========================================================================
    # STEP 6: Retrain Best Model and Plot ROC
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Final Model Training")
    print("="*80)
    
    ml_classifier.train_model(X_selected, y, model_type=best_model)
    final_cv_results = ml_classifier.cross_validate(X_selected, y, cv_method='loocv')
    
    # Plot ROC curve
    ml_classifier.plot_roc_curve(final_cv_results)
    
    # ========================================================================
    # STEP 7: Make Predictions on New Patients
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Testing on New Patients")
    print("="*80)
    
    # Test on new simulated patients
    print("\nTesting on healthy patient...")
    new_healthy_data = generate_simulated_data(duration=20, has_parkinsons=False)
    prediction_healthy = ml_classifier.predict(new_healthy_data)
    print(f"  Prediction: {prediction_healthy['prediction']}")
    print(f"  Probability (Parkinson's): {prediction_healthy['probability_parkinsons']:.2%}")
    print(f"  Confidence: {prediction_healthy['confidence']}")
    
    print("\nTesting on Parkinson's patient...")
    new_pd_data = generate_simulated_data(duration=20, has_parkinsons=True)
    prediction_pd = ml_classifier.predict(new_pd_data)
    print(f"  Prediction: {prediction_pd['prediction']}")
    print(f"  Probability (Parkinson's): {prediction_pd['probability_parkinsons']:.2%}")
    print(f"  Confidence: {prediction_pd['confidence']}")
    
    # ========================================================================
    # STEP 8: Save Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: Saving Model")
    print("="*80)
    
    ml_classifier.save_model('parkinsons_model.pkl')
    
    # ========================================================================
    # STEP 9: Demonstrate Loading and Using Saved Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 9: Loading Saved Model")
    print("="*80)
    
    # Create new classifier instance
    new_classifier = ParkinsonsMLClassifier(feature_extractor)
    new_classifier.load_model('parkinsons_model.pkl')
    
    # Make prediction with loaded model
    print("\nMaking prediction with loaded model...")
    test_data = generate_simulated_data(duration=20, has_parkinsons=True)
    prediction = new_classifier.predict(test_data)
    print(f"  Prediction: {prediction['prediction']}")
    print(f"  Probability: {prediction['probability_parkinsons']:.2%}")
    
    print("\n" + "="*80)
    print("COMPLETE PIPELINE DEMONSTRATION FINISHED")
    print("="*80)
    print("\nNEXT STEPS FOR REAL DEPLOYMENT:")
    print("1. Replace simulated data with real Colmi ring data")
    print("2. Collect data from diagnosed patients (with clinical validation)")
    print("3. Retrain model on real patient cohort")
    print("4. Perform external validation on independent test set")
    print("5. Consider regulatory requirements (FDA, CE marking)")
    print("="*80)