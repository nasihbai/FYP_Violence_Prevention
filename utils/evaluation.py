"""
Model Evaluation Framework
==========================
Comprehensive evaluation metrics and visualization for violence detection models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualization.

    Features:
    - Confusion matrix
    - Classification report
    - ROC curve and AUC
    - Precision-Recall curve
    - Per-class metrics
    - Threshold analysis
    """

    def __init__(
        self,
        class_names: List[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            class_names: List of class names
            output_dir: Directory to save evaluation results
        """
        self.class_names = class_names or ['neutral', 'violent']
        self.output_dir = Path(output_dir) if output_dir else Path('evaluation_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        save_results: bool = True
    ) -> Dict:
        """
        Perform comprehensive evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            save_results: Whether to save results to disk

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # Basic metrics
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        results['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        results['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        results['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        # Per-class metrics
        results['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        results['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        results['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        results['classification_report'] = report

        # ROC and AUC (for binary classification)
        if y_prob is not None and len(self.class_names) == 2:
            # Get probability for positive class
            if y_prob.ndim == 2:
                pos_prob = y_prob[:, 1]
            else:
                pos_prob = y_prob

            fpr, tpr, thresholds = roc_curve(y_true, pos_prob)
            roc_auc = auc(fpr, tpr)

            results['roc_auc'] = float(roc_auc)
            results['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }

            # Precision-Recall curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, pos_prob)
            results['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }

            # Find optimal threshold
            optimal_threshold = self._find_optimal_threshold(fpr, tpr, thresholds)
            results['optimal_threshold'] = float(optimal_threshold)

        self.results = results

        # Save results
        if save_results:
            self._save_results(results)

        return results

    def _find_optimal_threshold(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray
    ) -> float:
        """Find optimal threshold using Youden's J statistic."""
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize values
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            title='Confusion Matrix' + (' (Normalized)' if normalize else ''),
            ylabel='True Label',
            xlabel='Predicted Label'
        )

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        if save:
            filepath = self.output_dir / 'confusion_matrix.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {filepath}")

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if y_prob.ndim == 2:
            pos_prob = y_prob[:, 1]
        else:
            pos_prob = y_prob

        fpr, tpr, _ = roc_curve(y_true, pos_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save:
            filepath = self.output_dir / 'roc_curve.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to {filepath}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if y_prob.ndim == 2:
            pos_prob = y_prob[:, 1]
        else:
            pos_prob = y_prob

        precision, recall, _ = precision_recall_curve(y_true, pos_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)

        # Add F1 score contours
        f1_scores = np.linspace(0.2, 0.8, num=4)
        for f1 in f1_scores:
            x = np.linspace(0.01, 1)
            y = f1 * x / (2 * x - f1)
            ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.3, linestyle='--')
            ax.annotate(f'F1={f1:.1f}', xy=(0.9, y[len(y)//10]), fontsize=8, alpha=0.5)

        fig.tight_layout()

        if save:
            filepath = self.output_dir / 'precision_recall_curve.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {filepath}")

        return fig

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        Analyze metrics across different thresholds.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if y_prob.ndim == 2:
            pos_prob = y_prob[:, 1]
        else:
            pos_prob = y_prob

        thresholds = np.arange(0.1, 1.0, 0.05)
        precisions = []
        recalls = []
        f1s = []
        accuracies = []

        for thresh in thresholds:
            pred = (pos_prob >= thresh).astype(int)
            precisions.append(precision_score(y_true, pred, zero_division=0))
            recalls.append(recall_score(y_true, pred, zero_division=0))
            f1s.append(f1_score(y_true, pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, pred))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precisions, label='Precision', marker='o', markersize=3)
        ax.plot(thresholds, recalls, label='Recall', marker='s', markersize=3)
        ax.plot(thresholds, f1s, label='F1 Score', marker='^', markersize=3)
        ax.plot(thresholds, accuracies, label='Accuracy', marker='d', markersize=3)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Metrics vs. Classification Threshold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.95])
        ax.set_ylim([0, 1.05])

        # Mark optimal threshold
        optimal_idx = np.argmax(f1s)
        ax.axvline(x=thresholds[optimal_idx], color='red', linestyle='--', alpha=0.5,
                   label=f'Optimal (F1): {thresholds[optimal_idx]:.2f}')

        fig.tight_layout()

        if save:
            filepath = self.output_dir / 'threshold_analysis.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {filepath}")

        return fig

    def plot_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ):
        """Generate all evaluation plots."""
        self.plot_confusion_matrix(y_true, y_pred)

        if y_prob is not None:
            self.plot_roc_curve(y_true, y_prob)
            self.plot_precision_recall_curve(y_true, y_prob)
            self.plot_threshold_analysis(y_true, y_prob)

    def _save_results(self, results: Dict):
        """Save evaluation results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f'evaluation_results_{timestamp}.json'

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {filepath}")

    def print_summary(self):
        """Print evaluation summary."""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {self.results['accuracy']:.4f}")
        print(f"  Precision: {self.results['precision_macro']:.4f}")
        print(f"  Recall:    {self.results['recall_macro']:.4f}")
        print(f"  F1 Score:  {self.results['f1_macro']:.4f}")

        if 'roc_auc' in self.results:
            print(f"  ROC AUC:   {self.results['roc_auc']:.4f}")

        if 'optimal_threshold' in self.results:
            print(f"  Optimal Threshold: {self.results['optimal_threshold']:.4f}")

        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {self.results['precision_per_class'][i]:.4f}")
            print(f"    Recall:    {self.results['recall_per_class'][i]:.4f}")
            print(f"    F1 Score:  {self.results['f1_per_class'][i]:.4f}")

        print(f"\nConfusion Matrix:")
        cm = np.array(self.results['confusion_matrix'])
        print(f"  Predicted →")
        print(f"  True ↓     {' '.join([f'{c:>10}' for c in self.class_names])}")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:>10} {' '.join([f'{v:>10}' for v in cm[i]])}")

        print("\n" + "=" * 60)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = None,
    output_dir: str = 'evaluation_results'
) -> Dict:
    """
    Convenience function to evaluate a model.

    Args:
        model: Keras model
        X_test: Test features
        y_test: Test labels
        class_names: Class names
        output_dir: Output directory

    Returns:
        Evaluation results dictionary
    """
    # Get predictions
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Evaluate
    evaluator = ModelEvaluator(class_names=class_names, output_dir=output_dir)
    results = evaluator.evaluate(y_test, y_pred, y_prob)
    evaluator.plot_all(y_test, y_pred, y_prob)
    evaluator.print_summary()

    return results
