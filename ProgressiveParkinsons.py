"""
Progressive Micrographia Detection from Accelerometer Data
For use with smart ring devices (e.g., Colmi)

WARNING: This code is for algorithm development and testing only.
DO NOT train ML models on simulated data - use real patient data only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress, spearmanr
from typing import Dict, Tuple, List
import pandas as pd
from collections import Counter


class MicrographiaDetector:
    """
    Detects progressive micrographia from 3-axis accelerometer data
    """
    
    def __init__(self, sampling_rate: int = 100, segment_duration: float = 2.0):
        """
        Args:
            sampling_rate: Hz, typically 50-100 for smart rings
            segment_duration: seconds per segment for analysis
        """
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sampling_rate)
    
    def preprocess_signal(self, acc_data: np.ndarray) -> np.ndarray:
        """
        Preprocess accelerometer data
        
        Args:
            acc_data: (N, 3) array of X, Y, Z accelerations
            
        Returns:
            Filtered acceleration magnitude
        """
        # Calculate magnitude
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        
        # Remove DC offset (gravity)
        acc_magnitude = acc_magnitude - np.mean(acc_magnitude)
        
        # Bandpass filter to remove noise and drift
        # Keep 0.5-20 Hz (handwriting frequency range)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 20.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        acc_filtered = signal.filtfilt(b, a, acc_magnitude)
        
        return np.abs(acc_filtered)  # Absolute value for amplitude
    
    def segment_signal(self, acc_magnitude: np.ndarray) -> List[np.ndarray]:
        """
        Split signal into temporal segments
        
        Returns:
            List of signal segments
        """
        num_segments = len(acc_magnitude) // self.segment_samples
        segments = []
        
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segments.append(acc_magnitude[start:end])
        
        return segments
    
    def extract_amplitude_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Extract amplitude-related features from a segment
        
        Returns:
            Dictionary of features indicating writing size
        """
        features = {}
        
        # Basic amplitude measures
        features['mean_amplitude'] = np.mean(segment)
        features['median_amplitude'] = np.median(segment)
        features['max_amplitude'] = np.max(segment)
        features['std_amplitude'] = np.std(segment)
        features['amplitude_range'] = np.ptp(segment)  # Peak-to-peak
        
        # Percentiles (robust to outliers)
        features['p75_amplitude'] = np.percentile(segment, 75)
        features['p90_amplitude'] = np.percentile(segment, 90)
        features['p95_amplitude'] = np.percentile(segment, 95)
        
        # Peak characteristics
        peaks, properties = signal.find_peaks(segment, height=np.mean(segment))
        if len(peaks) > 0:
            features['mean_peak_height'] = np.mean(properties['peak_heights'])
            features['num_peaks'] = len(peaks)
        else:
            features['mean_peak_height'] = 0
            features['num_peaks'] = 0
        
        # Energy (related to movement extent)
        features['signal_energy'] = np.sum(segment ** 2)
        features['rms_amplitude'] = np.sqrt(np.mean(segment ** 2))
        
        return features
    
    def detect_progressive_reduction(
        self, 
        segment_features: List[Dict[str, float]], 
        feature_key: str = 'mean_amplitude'
    ) -> Dict[str, float]:
        """
        Analyze temporal trend in amplitude features
        
        Args:
            segment_features: List of feature dicts from each segment
            feature_key: Which feature to analyze for progression
            
        Returns:
            Dictionary with trend analysis results
        """
        # Extract feature values across segments
        values = np.array([f[feature_key] for f in segment_features])
        segment_indices = np.arange(len(values))
        
        # Linear regression: amplitude ~ time
        slope, intercept, r_value, p_value, std_err = linregress(
            segment_indices, values
        )
        
        # Spearman correlation (robust to non-linear trends)
        spearman_corr, spearman_p = spearmanr(segment_indices, values)
        
        # Percentage change from first to last segment
        if values[0] != 0:
            pct_change = 100 * (values[-1] - values[0]) / values[0]
        else:
            pct_change = 0
        
        # Compare first third vs last third
        first_third = values[:len(values)//3]
        last_third = values[-len(values)//3:]
        early_mean = np.mean(first_third)
        late_mean = np.mean(last_third)
        
        if early_mean != 0:
            early_late_pct_change = 100 * (late_mean - early_mean) / early_mean
        else:
            early_late_pct_change = 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(first_third)**2 + np.std(last_third)**2) / 2)
        if pooled_std > 0:
            cohens_d = (early_mean - late_mean) / pooled_std
        else:
            cohens_d = 0
        
        results = {
            'slope': slope,  # Negative = decreasing amplitude
            'slope_normalized': slope / (np.mean(values) + 1e-6),  # Normalized by mean
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'pct_change': pct_change,
            'early_late_pct_change': early_late_pct_change,
            'cohens_d': cohens_d,
            'segment_values': values.tolist()
        }
        
        return results
    
    def classify_micrographia(
        self, 
        trend_results: Dict[str, float],
        threshold_slope: float = -0.05,
        threshold_pct_change: float = -20.0,
        threshold_p_value: float = 0.05
    ) -> Dict[str, any]:
        """
        Classify whether progressive micrographia is present
        
        Args:
            trend_results: Output from detect_progressive_reduction
            threshold_slope: Negative slope threshold (normalized)
            threshold_pct_change: Percentage reduction threshold
            threshold_p_value: Statistical significance threshold
            
        Returns:
            Classification results with confidence
        """
        # Check multiple criteria
        has_negative_trend = trend_results['slope'] < 0
        significant_trend = trend_results['p_value'] < threshold_p_value
        large_reduction = trend_results['early_late_pct_change'] < threshold_pct_change
        strong_correlation = abs(trend_results['spearman_corr']) > 0.5
        
        # Count how many criteria are met
        criteria_met = sum([
            has_negative_trend,
            significant_trend,
            large_reduction,
            strong_correlation
        ])
        
        # Classification
        if criteria_met >= 3:
            classification = 'definite_micrographia'
            confidence = 'high'
        elif criteria_met == 2:
            classification = 'probable_micrographia'
            confidence = 'medium'
        elif criteria_met == 1:
            classification = 'possible_micrographia'
            confidence = 'low'
        else:
            classification = 'no_micrographia'
            confidence = 'high'
        
        return {
            'classification': classification,
            'confidence': confidence,
            'criteria_met': criteria_met,
            'criteria': {
                'has_negative_trend': has_negative_trend,
                'significant_trend': significant_trend,
                'large_reduction': large_reduction,
                'strong_correlation': strong_correlation
            },
            'metrics': {
                'slope': trend_results['slope'],
                'pct_change': trend_results['early_late_pct_change'],
                'p_value': trend_results['p_value'],
                'r_squared': trend_results['r_squared']
            }
        }
    
    def analyze_sample(
        self, 
        acc_data: np.ndarray,
        visualize: bool = True
    ) -> Dict[str, any]:
        """
        Complete analysis pipeline for one handwriting sample
        
        Args:
            acc_data: (N, 3) accelerometer data
            visualize: Whether to create plots
            
        Returns:
            Complete analysis results
        """
        # Step 1: Preprocess
        acc_magnitude = self.preprocess_signal(acc_data)
        
        # Step 2: Segment
        segments = self.segment_signal(acc_magnitude)
        
        if len(segments) < 3:
            return {
                'error': 'Insufficient data - need at least 3 segments',
                'num_segments': len(segments)
            }
        
        # Step 3: Extract features per segment
        segment_features = [
            self.extract_amplitude_features(seg) for seg in segments
        ]
        
        # Step 4: Analyze trends for multiple features
        results = {}
        important_features = [
            'mean_amplitude', 'p90_amplitude', 'rms_amplitude', 
            'max_amplitude', 'signal_energy'
        ]
        
        for feature in important_features:
            trend = self.detect_progressive_reduction(segment_features, feature)
            classification = self.classify_micrographia(trend)
            results[feature] = {
                'trend': trend,
                'classification': classification
            }
        
        # Step 5: Aggregate classification (vote across features)
        classifications = [results[f]['classification']['classification'] 
                          for f in important_features]
        
        # Most common classification
        vote_counts = Counter(classifications)
        final_classification = vote_counts.most_common(1)[0][0]
        
        # Confidence based on agreement
        agreement = vote_counts[final_classification] / len(classifications)
        
        # Step 6: Visualization
        if visualize:
            self._visualize_analysis(
                acc_magnitude, segments, segment_features, 
                results, final_classification
            )
        
        return {
            'final_classification': final_classification,
            'agreement': agreement,
            'feature_results': results,
            'num_segments': len(segments),
            'total_duration': len(acc_data) / self.sampling_rate
        }
    
    def _visualize_analysis(
        self,
        acc_magnitude: np.ndarray,
        segments: List[np.ndarray],
        segment_features: List[Dict],
        results: Dict,
        final_classification: str
    ):
        """
        Create comprehensive visualization of analysis
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Progressive Micrographia Analysis\nClassification: {final_classification}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Raw signal with segment boundaries
        ax = axes[0, 0]
        time = np.arange(len(acc_magnitude)) / self.sampling_rate
        ax.plot(time, acc_magnitude, 'b-', alpha=0.7, linewidth=0.5)
        
        # Mark segment boundaries
        for i in range(len(segments)):
            segment_time = (i * self.segment_samples) / self.sampling_rate
            ax.axvline(segment_time, color='r', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration Magnitude')
        ax.set_title('Signal with Segment Boundaries')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean amplitude across segments
        ax = axes[0, 1]
        segment_nums = np.arange(len(segment_features))
        mean_amps = [f['mean_amplitude'] for f in segment_features]
        
        ax.plot(segment_nums, mean_amps, 'o-', linewidth=2, markersize=8)
        
        # Add trend line
        z = np.polyfit(segment_nums, mean_amps, 1)
        p = np.poly1d(z)
        ax.plot(segment_nums, p(segment_nums), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend (slope={z[0]:.3f})')
        
        ax.set_xlabel('Segment Number')
        ax.set_ylabel('Mean Amplitude')
        ax.set_title('Amplitude Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Multiple amplitude features
        ax = axes[1, 0]
        features_to_plot = ['mean_amplitude', 'p90_amplitude', 'max_amplitude']
        for feat in features_to_plot:
            values = [f[feat] for f in segment_features]
            # Normalize to 0-1 for comparison
            values_norm = (values - np.min(values)) / (np.ptp(values) + 1e-6)
            ax.plot(segment_nums, values_norm, 'o-', label=feat, linewidth=2, markersize=6)
        
        ax.set_xlabel('Segment Number')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title('Multiple Amplitude Features (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistical metrics
        ax = axes[1, 1]
        feature_names = list(results.keys())
        slopes = [results[f]['trend']['slope_normalized'] for f in feature_names]
        p_values = [results[f]['trend']['p_value'] for f in feature_names]
        
        x = np.arange(len(feature_names))
        bars = ax.bar(x, slopes, alpha=0.7)
        
        # Color bars by p-value significance
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.01:
                bar.set_color('darkred')
            elif p_val < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('gray')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, 
                   label='Threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Normalized Slope')
        ax.set_title('Trend Slopes by Feature\n(red: p<0.01, orange: p<0.05)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Percentage changes
        ax = axes[2, 0]
        pct_changes = [results[f]['trend']['early_late_pct_change'] for f in feature_names]
        bars = ax.barh(feature_names, pct_changes, alpha=0.7)
        
        # Color by magnitude
        for bar, pct in zip(bars, pct_changes):
            if pct < -30:
                bar.set_color('darkred')
            elif pct < -20:
                bar.set_color('orange')
            elif pct < -10:
                bar.set_color('yellow')
            else:
                bar.set_color('lightgreen')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(-20, color='red', linestyle='--', linewidth=1, alpha=0.5, 
                   label='Threshold (-20%)')
        ax.set_xlabel('% Change (Early to Late)')
        ax.set_title('Amplitude Reduction (%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 6: Summary table
        ax = axes[2, 1]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = []
        for feature in feature_names:
            cls = results[feature]['classification']
            summary_data.append([
                feature,
                cls['classification'],
                f"{cls['metrics']['pct_change']:.1f}%",
                f"{cls['metrics']['p_value']:.3f}",
                f"{cls['metrics']['r_squared']:.3f}"
            ])
        
        table = ax.table(
            cellText=summary_data,
            colLabels=['Feature', 'Classification', '% Change', 'p-value', 'R²'],
            cellLoc='left',
            loc='center',
            colWidths=[0.25, 0.25, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color-code classifications
        for i, row in enumerate(summary_data, 1):
            cls = row[1]
            if 'definite' in cls:
                color = '#ffcccc'
            elif 'probable' in cls:
                color = '#ffe6cc'
            elif 'possible' in cls:
                color = '#ffffcc'
            else:
                color = '#ccffcc'
            table[(i, 1)].set_facecolor(color)
        
        plt.tight_layout()
        plt.show()


class ParkinsonsFeaturesExtractor:
    """
    Comprehensive feature extraction for Parkinson's detection
    """
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.micrographia_detector = MicrographiaDetector(sampling_rate)
    
    def extract_all_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive feature set for Parkinson's
        """
        features = {}
        
        # 1. Micrographia features
        micro_results = self.micrographia_detector.analyze_sample(
            acc_data, visualize=False
        )
        features['micrographia_classification'] = micro_results['final_classification']
        features['micrographia_pct_change'] = micro_results['feature_results']['mean_amplitude']['trend']['early_late_pct_change']
        features['micrographia_slope'] = micro_results['feature_results']['mean_amplitude']['trend']['slope']
        
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
        
        return features
    
    def _extract_tremor_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract tremor-specific features (Parkinson's: 4-6 Hz)"""
        from scipy.signal import welch
        
        features = {}
        
        for axis, axis_name in enumerate(['x', 'y', 'z']):
            freqs, psd = welch(acc_data[:, axis], fs=self.sampling_rate, nperseg=256)
            
            # Parkinson's tremor band (4-6 Hz)
            tremor_band = (freqs >= 4) & (freqs <= 6)
            tremor_power = np.sum(psd[tremor_band])
            total_power = np.sum(psd)
            
            features[f'{axis_name}_tremor_power'] = tremor_power
            features[f'{axis_name}_tremor_ratio'] = tremor_power / (total_power + 1e-6)
            
            # Dominant frequency in tremor band
            if np.any(tremor_band):
                tremor_freqs = freqs[tremor_band]
                tremor_psd = psd[tremor_band]
                features[f'{axis_name}_tremor_peak_freq'] = tremor_freqs[np.argmax(tremor_psd)]
        
        return features
    
    def _extract_velocity_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """Extract velocity-related features"""
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        
        # Approximate velocity (integration of acceleration)
        velocity = np.cumsum(acc_magnitude) / self.sampling_rate
        velocity = velocity - np.mean(velocity)  # Remove drift
        
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
        
        # Find peaks (stroke cycles)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acc_magnitude, distance=self.sampling_rate//10)
        
        if len(peaks) > 1:
            # Inter-peak intervals
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
# UTILITY FUNCTIONS
# ============================================================================

def generate_simulated_data(duration=20, sampling_rate=100, micrographia=True):
    """
    Generate synthetic accelerometer data
    
    WARNING: Only for algorithm testing, NOT for training ML models!
    Simulated micrographia has different kinematic signatures than real pathological micrographia.
    
    Args:
        duration: seconds
        sampling_rate: Hz
        micrographia: if True, amplitude decreases over time
        
    Returns:
        (N, 3) array of accelerometer data
    """
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Base writing signal (oscillations)
    base_freq = 2.0  # Hz, typical handwriting frequency
    
    if micrographia:
        # Amplitude decreases linearly
        amplitude = np.linspace(1.0, 0.3, len(t))  # 70% reduction
    else:
        # Constant amplitude
        amplitude = np.ones(len(t))
    
    # Simulate writing strokes
    signal_x = amplitude * np.sin(2 * np.pi * base_freq * t)
    signal_y = amplitude * np.cos(2 * np.pi * base_freq * 1.5 * t)
    signal_z = amplitude * 0.5 * np.sin(2 * np.pi * base_freq * 0.8 * t)
    
    # Add noise
    noise = np.random.normal(0, 0.1, (len(t), 3))
    
    acc_data = np.column_stack([signal_x, signal_y, signal_z]) + noise
    
    return acc_data


def load_colmi_data(filepath):
    """
    Load accelerometer data from Colmi ring Bluetooth capture
    
    Adjust this based on your actual data format
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        (N, 3) array of accelerometer data
    """
    # PLACEHOLDER - adjust to your actual format
    df = pd.read_csv(filepath)
    
    # Assumes columns are named 'acc_x', 'acc_y', 'acc_z'
    # Adjust column names as needed
    acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
    
    return acc_data


def analyze_patient_cohort(patient_files):
    """
    Analyze micrographia across multiple patients
    
    Args:
        patient_files: dict mapping patient_id to filepath
        
    Returns:
        DataFrame with summary results
    """
    detector = MicrographiaDetector()
    results_summary = []
    
    for patient_id, filepath in patient_files.items():
        print(f"Analyzing {patient_id}...")
        
        acc_data = load_colmi_data(filepath)
        results = detector.analyze_sample(acc_data, visualize=False)
        
        # Extract key metrics
        mean_amp_results = results['feature_results']['mean_amplitude']
        
        results_summary.append({
            'patient_id': patient_id,
            'classification': results['final_classification'],
            'agreement': results['agreement'],
            'pct_change': mean_amp_results['trend']['early_late_pct_change'],
            'p_value': mean_amp_results['trend']['p_value'],
            'slope': mean_amp_results['trend']['slope'],
            'r_squared': mean_amp_results['trend']['r_squared']
        })
    
    return pd.DataFrame(results_summary)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PROGRESSIVE MICROGRAPHIA DETECTION - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Test on simulated data (algorithm validation only!)
    print("\n1. Testing on SIMULATED micrographia data")
    print("-" * 80)
    print("WARNING: This is for algorithm testing ONLY.")
    print("Do NOT use simulated data to train ML models!\n")
    
    sim_data_micro = generate_simulated_data(duration=20, micrographia=True)
    detector = MicrographiaDetector(sampling_rate=100, segment_duration=2.0)
    results_micro = detector.analyze_sample(sim_data_micro, visualize=True)
    
    print(f"\nFinal Classification: {results_micro['final_classification']}")
    print(f"Agreement across features: {results_micro['agreement']:.2%}")
    print(f"Number of segments analyzed: {results_micro['num_segments']}")
    print(f"Total duration: {results_micro['total_duration']:.1f}s")
    
    # Example 2: Test on simulated normal writing
    print("\n2. Testing on SIMULATED normal writing")
    print("-" * 80)
    
    sim_data_normal = generate_simulated_data(duration=20, micrographia=False)
    results_normal = detector.analyze_sample(sim_data_normal, visualize=True)
    
    print(f"\nFinal Classification: {results_normal['final_classification']}")
    print(f"Agreement across features: {results_normal['agreement']:.2%}")
    
    # Example 3: Extract comprehensive Parkinson's features
    print("\n3. Extracting comprehensive Parkinson's features")
    print("-" * 80)
    
    extractor = ParkinsonsFeaturesExtractor(sampling_rate=100)
    all_features = extractor.extract_all_features(sim_data_micro)
    
    print("\nExtracted features:")
    for key, value in list(all_features.items())[:10]:  # Show first 10
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"  ... ({len(all_features)} total features)")
    
    # Example 4: Batch analysis (commented out - requires real data)
    print("\n4. Batch analysis example (requires real data files)")
    print("-" * 80)
    print("Uncomment the following code when you have real patient data:\n")
    print("""
        patient_files = {
            'PD_001': 'data/pd_patient_001.csv',
            'PD_002': 'data/pd_patient_002.csv',
            'HC_001': 'data/healthy_control_001.csv',
            'HC_002': 'data/healthy_control_002.csv',
        }
    
    summary_df = analyze_patient_cohort(patient_files)
    print(summary_df)
    
    # Statistical comparison
    from scipy.stats import ttest_ind
    pd_patients = summary_df[summary_df['patient_id'].str.startswith('PD')]
    hc_patients = summary_df[summary_df['patient_id'].str.startswith('HC')]
    
    t_stat, p_val = ttest_ind(pd_patients['pct_change'], hc_patients['pct_change'])
    print(f"PD vs HC % Change: t={t_stat:.2f}, p={p_val:.4f}")
    """)