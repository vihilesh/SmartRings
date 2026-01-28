"""
Advanced Visualization Module for Parkinson's Detection System

This module provides publication-quality visualizations for:
- Feature analysis
- Model performance
- Patient comparisons
- Temporal trends
- Clinical interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


class ParkinsonsVisualizer:
    """
    Advanced visualization suite for Parkinson's detection
    """
    
    def __init__(self, color_healthy='#2ecc71', color_pd='#e74c3c'):
        self.color_healthy = color_healthy
        self.color_pd = color_pd
        
    def plot_signal_decomposition(self, acc_data, sampling_rate=100, patient_label="Patient"):
        """
        Comprehensive signal decomposition visualization
        Shows raw signal, preprocessing steps, and feature extraction
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Calculate derived signals
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        acc_centered = acc_magnitude - np.mean(acc_magnitude)
        time = np.arange(len(acc_data)) / sampling_rate
        
        # 1. Raw 3-axis accelerometer data
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, acc_data[:, 0], label='X-axis', alpha=0.7, linewidth=1)
        ax1.plot(time, acc_data[:, 1], label='Y-axis', alpha=0.7, linewidth=1)
        ax1.plot(time, acc_data[:, 2], label='Z-axis', alpha=0.7, linewidth=1)
        ax1.set_ylabel('Acceleration (m/s²)', fontsize=11, fontweight='bold')
        ax1.set_title('Raw 3-Axis Accelerometer Data', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Magnitude (with gravity)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time, acc_magnitude, color='#3498db', linewidth=1.5)
        ax2.axhline(np.mean(acc_magnitude), color='#e74c3c', linestyle='--', 
                   linewidth=2, label=f'Mean (Gravity) = {np.mean(acc_magnitude):.2f} m/s²')
        ax2.fill_between(time, acc_magnitude, np.mean(acc_magnitude), 
                         alpha=0.2, color='#3498db')
        ax2.set_ylabel('Magnitude (m/s²)', fontsize=11, fontweight='bold')
        ax2.set_title('Magnitude (With Gravity Component)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Magnitude (gravity removed)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(time, acc_centered, color='#2ecc71', linewidth=1.5)
        ax3.axhline(0, color='#95a5a6', linestyle='-', linewidth=1, alpha=0.5)
        ax3.fill_between(time, acc_centered, 0, alpha=0.3, color='#2ecc71')
        ax3.set_ylabel('Magnitude (m/s²)', fontsize=11, fontweight='bold')
        ax3.set_title('Magnitude (Gravity Removed)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Frequency spectrum (tremor analysis)
        ax4 = fig.add_subplot(gs[2, 0])
        from scipy.signal import welch
        freqs, psd = welch(acc_centered, fs=sampling_rate, nperseg=min(256, len(acc_centered)))
        ax4.semilogy(freqs, psd, color='#9b59b6', linewidth=2)
        
        # Highlight tremor band (4-6 Hz)
        tremor_band = (freqs >= 4) & (freqs <= 6)
        ax4.fill_between(freqs[tremor_band], psd[tremor_band], alpha=0.3, 
                        color='#e74c3c', label='Tremor Band (4-6 Hz)')
        ax4.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Power Spectral Density', fontsize=11, fontweight='bold')
        ax4.set_title('Frequency Analysis (Tremor Detection)', fontsize=12, fontweight='bold')
        ax4.set_xlim([0, 20])
        ax4.legend(loc='upper right', framealpha=0.9)
        ax4.grid(True, alpha=0.3, which='both')
        
        # 5. Time-domain features
        ax5 = fig.add_subplot(gs[2, 1])
        window_size = int(2 * sampling_rate)  # 2-second windows
        num_windows = len(acc_centered) // window_size
        
        window_energies = []
        window_times = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = acc_centered[start:end]
            energy = np.sum(window ** 2)
            window_energies.append(energy)
            window_times.append((start + end) / 2 / sampling_rate)
        
        ax5.plot(window_times, window_energies, 'o-', color='#e67e22', 
                markersize=8, linewidth=2, label='Window Energy')
        
        # Trend line
        if len(window_times) > 2:
            z = np.polyfit(window_times, window_energies, 1)
            p = np.poly1d(z)
            ax5.plot(window_times, p(window_times), '--', color='#c0392b', 
                    linewidth=2, label=f'Trend (slope={z[0]:.2f})')
        
        ax5.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Signal Energy', fontsize=11, fontweight='bold')
        ax5.set_title('Temporal Energy Evolution', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistical summary
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Calculate statistics
        stats_data = {
            'Metric': [
                'Mean Amplitude',
                'Std Deviation',
                'Max Amplitude',
                'Energy',
                'Tremor Power (4-6 Hz)',
                'Dominant Freq (Hz)'
            ],
            'Value': [
                f'{np.mean(np.abs(acc_centered)):.3f} m/s²',
                f'{np.std(acc_centered):.3f} m/s²',
                f'{np.max(np.abs(acc_centered)):.3f} m/s²',
                f'{np.sum(acc_centered**2):.2f}',
                f'{np.sum(psd[tremor_band]):.2e}',
                f'{freqs[np.argmax(psd)]:.2f} Hz'
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create table
        table = ax6.table(cellText=stats_df.values,
                         colLabels=stats_df.columns,
                         cellLoc='left',
                         loc='center',
                         bbox=[0.1, 0.2, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(stats_df) + 1):
            color = '#ecf0f1' if i % 2 == 0 else 'white'
            for j in range(len(stats_df.columns)):
                table[(i, j)].set_facecolor(color)
        
        ax6.set_title('Feature Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle(f'Signal Decomposition Analysis - {patient_label}', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_comparison(self, features_healthy, features_pd, feature_names=None):
        """
        Comprehensive feature comparison between healthy and PD patients
        Using violin plots, box plots, and statistical tests
        """
        if feature_names is None:
            feature_names = list(features_healthy.keys())
        
        # Select top features for visualization (avoid clutter)
        if len(feature_names) > 12:
            # Calculate effect sizes and select top 12
            effect_sizes = []
            for feat in feature_names:
                healthy_vals = [f[feat] for f in features_healthy if feat in f]
                pd_vals = [f[feat] for f in features_pd if feat in f]
                
                if len(healthy_vals) > 0 and len(pd_vals) > 0:
                    mean_diff = abs(np.mean(pd_vals) - np.mean(healthy_vals))
                    pooled_std = np.sqrt((np.std(healthy_vals)**2 + np.std(pd_vals)**2) / 2)
                    cohen_d = mean_diff / (pooled_std + 1e-6)
                    effect_sizes.append((feat, cohen_d))
            
            effect_sizes.sort(key=lambda x: x[1], reverse=True)
            feature_names = [f[0] for f in effect_sizes[:12]]
        
        # Prepare data
        data_list = []
        for feat in feature_names:
            for f in features_healthy:
                if feat in f:
                    data_list.append({'Feature': feat, 'Value': f[feat], 'Group': 'Healthy'})
            for f in features_pd:
                if feat in f:
                    data_list.append({'Feature': feat, 'Value': f[feat], 'Group': 'Parkinson\'s'})
        
        df = pd.DataFrame(data_list)
        
        # Create figure
        n_features = len(feature_names)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feat in enumerate(feature_names):
            ax = axes[idx]
            feat_data = df[df['Feature'] == feat]
            
            # Violin plot with box plot overlay
            parts = ax.violinplot(
                [feat_data[feat_data['Group']=='Healthy']['Value'].values,
                 feat_data[feat_data['Group']=='Parkinson\'s']['Value'].values],
                positions=[0, 1],
                widths=0.7,
                showmeans=True,
                showmedians=True
            )
            
            # Color violins
            for i, pc in enumerate(parts['bodies']):
                color = self.color_healthy if i == 0 else self.color_pd
                pc.set_facecolor(color)
                pc.set_alpha(0.3)
            
            # Add box plots
            bp = ax.boxplot(
                [feat_data[feat_data['Group']=='Healthy']['Value'].values,
                 feat_data[feat_data['Group']=='Parkinson\'s']['Value'].values],
                positions=[0, 1],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2, color='black'),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )
            
            # Color boxes
            bp['boxes'][0].set_facecolor(self.color_healthy)
            bp['boxes'][0].set_alpha(0.5)
            bp['boxes'][1].set_facecolor(self.color_pd)
            bp['boxes'][1].set_alpha(0.5)
            
            # Statistical test (t-test)
            from scipy.stats import ttest_ind
            healthy_vals = feat_data[feat_data['Group']=='Healthy']['Value'].values
            pd_vals = feat_data[feat_data['Group']=='Parkinson\'s']['Value'].values
            
            if len(healthy_vals) > 1 and len(pd_vals) > 1:
                t_stat, p_val = ttest_ind(healthy_vals, pd_vals)
                
                # Add significance annotation
                y_max = max(feat_data['Value'].max(), feat_data['Value'].max())
                y_min = min(feat_data['Value'].min(), feat_data['Value'].min())
                y_range = y_max - y_min
                
                sig_y = y_max + 0.1 * y_range
                
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                ax.plot([0, 1], [sig_y, sig_y], 'k-', linewidth=1.5)
                ax.text(0.5, sig_y + 0.02 * y_range, sig_text, 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Add p-value
                ax.text(0.5, sig_y + 0.08 * y_range, f'p={p_val:.4f}', 
                       ha='center', va='bottom', fontsize=9)
            
            # Formatting
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Healthy', 'PD'], fontsize=10)
            ax.set_ylabel('Value', fontsize=10, fontweight='bold')
            ax.set_title(feat.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Comparison: Healthy vs. Parkinson\'s Disease\n*** p<0.001, ** p<0.01, * p<0.05, ns=not significant', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_patient_trajectory(self, patient_samples_over_time, timestamps, patient_id, diagnosis):
        """
        Visualize how a patient's handwriting features change over time
        Useful for tracking disease progression or medication effects
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        color = self.color_pd if diagnosis == 'Parkinson\'s' else self.color_healthy
        
        # Extract features over time
        features_over_time = []
        for sample in patient_samples_over_time:
            # Assuming sample is accelerometer data
            acc_magnitude = np.linalg.norm(sample, axis=1)
            acc_centered = acc_magnitude - np.mean(acc_magnitude)
            
            features = {
                'mean_amplitude': np.mean(np.abs(acc_centered)),
                'energy': np.sum(acc_centered ** 2),
                'max_amplitude': np.max(np.abs(acc_centered)),
                'std': np.std(acc_centered)
            }
            features_over_time.append(features)
        
        df = pd.DataFrame(features_over_time)
        df['timestamp'] = timestamps
        
        # Plot 1: Mean amplitude over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['timestamp'], df['mean_amplitude'], 'o-', color=color, 
                linewidth=2, markersize=8, alpha=0.7)
        z = np.polyfit(range(len(df)), df['mean_amplitude'], 1)
        p = np.poly1d(z)
        ax1.plot(df['timestamp'], p(range(len(df))), '--', color='black', 
                linewidth=2, alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
        ax1.set_ylabel('Mean Amplitude', fontsize=11, fontweight='bold')
        ax1.set_title('Mean Amplitude Progression', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['timestamp'], df['energy'], 'o-', color=color, 
                linewidth=2, markersize=8, alpha=0.7)
        z = np.polyfit(range(len(df)), df['energy'], 1)
        p = np.poly1d(z)
        ax2.plot(df['timestamp'], p(range(len(df))), '--', color='black', 
                linewidth=2, alpha=0.5, label=f'Trend (slope={z[0]:.2f})')
        ax2.set_ylabel('Signal Energy', fontsize=11, fontweight='bold')
        ax2.set_title('Signal Energy Progression', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Max amplitude over time
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df['timestamp'], df['max_amplitude'], 'o-', color=color, 
                linewidth=2, markersize=8, alpha=0.7)
        z = np.polyfit(range(len(df)), df['max_amplitude'], 1)
        p = np.poly1d(z)
        ax3.plot(df['timestamp'], p(range(len(df))), '--', color='black', 
                linewidth=2, alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
        ax3.set_ylabel('Max Amplitude', fontsize=11, fontweight='bold')
        ax3.set_title('Peak Movement Progression', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Variability over time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df['timestamp'], df['std'], 'o-', color=color, 
                linewidth=2, markersize=8, alpha=0.7)
        z = np.polyfit(range(len(df)), df['std'], 1)
        p = np.poly1d(z)
        ax4.plot(df['timestamp'], p(range(len(df))), '--', color='black', 
                linewidth=2, alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
        ax4.set_ylabel('Std Deviation', fontsize=11, fontweight='bold')
        ax4.set_title('Movement Variability', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: All features normalized on same plot
        ax5 = fig.add_subplot(gs[2, :])
        for col in ['mean_amplitude', 'energy', 'max_amplitude', 'std']:
            normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)
            ax5.plot(df['timestamp'], normalized, 'o-', label=col.replace('_', ' ').title(), 
                    linewidth=2, markersize=6, alpha=0.7)
        
        ax5.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
        ax5.set_title('All Features Normalized', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f'Longitudinal Trajectory - Patient {patient_id} ({diagnosis})', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_model_performance_dashboard(self, cv_results, model_name='Model'):
        """
        Comprehensive model performance visualization
        ROC, Precision-Recall, Confusion Matrix, and more
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        y_true = cv_results['y_true']
        y_pred = cv_results['y_pred']
        y_pred_proba = cv_results['y_pred_proba']
        
        # Plot 1: ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='#e74c3c', linewidth=3, label=f'ROC (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax1.fill_between(fpr, tpr, alpha=0.2, color='#e74c3c')
        ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 1])
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        ax2.plot(recall, precision, color='#3498db', linewidth=3, 
                label=f'PR (AP = {avg_precision:.3f})')
        ax2.fill_between(recall, precision, alpha=0.2, color='#3498db')
        ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower left', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix
        ax3 = fig.add_subplot(gs[0, 2])
        cm = cv_results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
                   xticklabels=['Healthy', 'PD'],
                   yticklabels=['Healthy', 'PD'],
                   cbar_kws={'label': 'Count'},
                   ax=ax3, linewidths=2, linecolor='black',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Plot 4: Probability Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        
        healthy_probs = y_pred_proba[y_true == 0]
        pd_probs = y_pred_proba[y_true == 1]
        
        ax4.hist(healthy_probs, bins=20, alpha=0.6, color=self.color_healthy, 
                label='Healthy', edgecolor='black', linewidth=1.5)
        ax4.hist(pd_probs, bins=20, alpha=0.6, color=self.color_pd, 
                label='Parkinson\'s', edgecolor='black', linewidth=1.5)
        ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, 
                   label='Decision Threshold')
        
        ax4.set_xlabel('Predicted Probability (PD)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
        ax4.legend(framealpha=0.9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Calibration Curve
        ax5 = fig.add_subplot(gs[1, 1])
        from sklearn.calibration import calibration_curve
        
        if len(np.unique(y_true)) > 1:
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
            ax5.plot(prob_pred, prob_true, 's-', color='#9b59b6', linewidth=2, 
                    markersize=8, label='Model')
            ax5.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly Calibrated')
            
            ax5.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Fraction of Positives', fontsize=11, fontweight='bold')
            ax5.set_title('Calibration Curve', fontsize=12, fontweight='bold')
            ax5.legend(framealpha=0.9)
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Metrics Summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Avg Precision'],
            'Value': [
                f'{accuracy_score(y_true, y_pred):.3f}',
                f'{precision_score(y_true, y_pred):.3f}',
                f'{recall_score(y_true, y_pred):.3f}',
                f'{f1_score(y_true, y_pred):.3f}',
                f'{roc_auc:.3f}',
                f'{avg_precision:.3f}'
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        table = ax6.table(cellText=metrics_df.values,
                         colLabels=metrics_df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.2, 0.8, 0.7])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(metrics_df) + 1):
            color = '#ecf0f1' if i % 2 == 0 else 'white'
            for j in range(len(metrics_df.columns)):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_text_props(weight='bold')
        
        ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle(f'{model_name} - Performance Dashboard', 
                    fontsize=15, fontweight='bold')
        
        return fig