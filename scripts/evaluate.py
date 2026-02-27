"""
BSL Translation System - Evaluation Metrics

Comprehensive evaluation for all pipeline components:
1. Speech-to-Text (Whisper): WER, CER
2. Text-to-Gloss: Accuracy, Coverage, BLEU
3. Gloss-to-Text: BLEU, ROUGE, Semantic Similarity
4. Sign Recognition: Accuracy, Top-K, F1, Confusion Matrix
5. End-to-End: Full pipeline metrics

Usage:
    python scripts/evaluate.py --all
    python scripts/evaluate.py --component recognition
    python scripts/evaluate.py --component gloss2text
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "inference"))
sys.path.insert(0, str(project_root / "scripts"))


# ============================================================
# Metric Calculations
# ============================================================

def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(
                    d[i-1, j] + 1,      # deletion
                    d[i, j-1] + 1,      # insertion
                    d[i-1, j-1] + 1     # substitution
                )
    
    return d[len(ref_words), len(hyp_words)] / max(len(ref_words), 1)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER)."""
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + 1)
    
    return d[len(ref_chars), len(hyp_chars)] / max(len(ref_chars), 1)


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU score (1-4 gram).
    
    Returns dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4, and combined BLEU.
    """
    from collections import Counter
    import math
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    scores = {}
    precisions = []
    
    for n in range(1, max_n + 1):
        # Get n-grams
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        
        # Count matches
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        
        precision = matches / max(total, 1)
        precisions.append(precision)
        scores[f'BLEU-{n}'] = precision
    
    # Brevity penalty
    bp = min(1, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    
    # Combined BLEU (geometric mean)
    if all(p > 0 for p in precisions):
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        bleu = 0.0
    
    scores['BLEU'] = bleu
    return scores


def rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L scores."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # ROUGE-1 (unigram overlap)
    ref_unigrams = set(ref_tokens)
    hyp_unigrams = set(hyp_tokens)
    overlap_1 = len(ref_unigrams & hyp_unigrams)
    rouge_1 = overlap_1 / max(len(ref_unigrams), 1)
    
    # ROUGE-2 (bigram overlap)
    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
    hyp_bigrams = set(zip(hyp_tokens[:-1], hyp_tokens[1:]))
    overlap_2 = len(ref_bigrams & hyp_bigrams)
    rouge_2 = overlap_2 / max(len(ref_bigrams), 1)
    
    # ROUGE-L (longest common subsequence)
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs = lcs_length(ref_tokens, hyp_tokens)
    rouge_l = lcs / max(len(ref_tokens), 1)
    
    return {
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l
    }


def accuracy_metrics(y_true: List, y_pred: List, top_k_preds: List[List] = None) -> Dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (top-1)
        top_k_preds: Optional list of top-k predictions per sample
    """
    from collections import Counter
    
    n = len(y_true)
    
    # Top-1 accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    top1_acc = correct / max(n, 1)
    
    # Top-5 accuracy (if provided)
    top5_acc = None
    if top_k_preds:
        top5_correct = sum(1 for t, preds in zip(y_true, top_k_preds) if t in preds[:5])
        top5_acc = top5_correct / max(n, 1)
    
    # Per-class metrics
    classes = sorted(set(y_true))
    class_metrics = {}
    
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for t in y_true if t == cls)
        }
    
    # Macro averages
    macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    
    return {
        'accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'num_samples': n,
        'num_classes': len(classes),
        'per_class': class_metrics
    }


def confusion_matrix(y_true: List, y_pred: List, labels: List = None) -> np.ndarray:
    """Build confusion matrix."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    
    return cm


# ============================================================
# Component Evaluators
# ============================================================

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    component: str
    metrics: Dict
    timestamp: str
    num_samples: int
    inference_time_ms: float = 0.0


class TextToGlossEvaluator:
    """Evaluate Text-to-Gloss conversion."""
    
    def __init__(self, vocabulary_path: str = None):
        from speech_to_bsl import TextToGloss
        
        vocab_path = vocabulary_path or str(project_root / "data/processed/vocabulary_extended.json")
        self.converter = TextToGloss(mode="simple", vocabulary_path=vocab_path)
    
    def evaluate(self, test_pairs: List[Tuple[str, List[str]]]) -> EvaluationResult:
        """
        Evaluate on test pairs of (english_text, expected_glosses).
        """
        results = {
            'exact_match': 0,
            'token_accuracy': [],
            'coverage': [],
            'bleu_scores': []
        }
        
        total_time = 0
        
        for text, expected_glosses in test_pairs:
            start = time.time()
            predicted = self.converter.convert(text)
            total_time += (time.time() - start) * 1000
            
            # Exact match
            if predicted == expected_glosses:
                results['exact_match'] += 1
            
            # Token accuracy
            correct = sum(1 for p, e in zip(predicted, expected_glosses) if p == e)
            results['token_accuracy'].append(correct / max(len(expected_glosses), 1))
            
            # Coverage
            info = self.converter.convert_with_info(text)
            results['coverage'].append(info['coverage'])
            
            # BLEU
            bleu = bleu_score(' '.join(expected_glosses), ' '.join(predicted))
            results['bleu_scores'].append(bleu['BLEU'])
        
        metrics = {
            'exact_match_rate': results['exact_match'] / len(test_pairs),
            'avg_token_accuracy': np.mean(results['token_accuracy']),
            'avg_coverage': np.mean(results['coverage']),
            'avg_bleu': np.mean(results['bleu_scores'])
        }
        
        return EvaluationResult(
            component='text_to_gloss',
            metrics=metrics,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            num_samples=len(test_pairs),
            inference_time_ms=total_time / len(test_pairs)
        )


class GlossToTextEvaluator:
    """Evaluate Gloss-to-Text conversion."""
    
    def __init__(self, mode: str = "groq"):
        from gloss_to_text import GlossToText
        self.converter = GlossToText(mode=mode)
    
    def evaluate(self, test_pairs: List[Tuple[List[str], str]]) -> EvaluationResult:
        """
        Evaluate on test pairs of (glosses, expected_english).
        """
        results = {
            'bleu': [],
            'rouge_1': [],
            'rouge_l': [],
            'wer': []
        }
        
        total_time = 0
        
        for glosses, expected_text in test_pairs:
            start = time.time()
            predicted = self.converter.convert(glosses)
            total_time += (time.time() - start) * 1000
            
            # BLEU
            bleu = bleu_score(expected_text, predicted)
            results['bleu'].append(bleu['BLEU'])
            
            # ROUGE
            rouge = rouge_scores(expected_text, predicted)
            results['rouge_1'].append(rouge['ROUGE-1'])
            results['rouge_l'].append(rouge['ROUGE-L'])
            
            # WER
            wer = word_error_rate(expected_text, predicted)
            results['wer'].append(wer)
        
        metrics = {
            'avg_bleu': np.mean(results['bleu']),
            'avg_rouge_1': np.mean(results['rouge_1']),
            'avg_rouge_l': np.mean(results['rouge_l']),
            'avg_wer': np.mean(results['wer'])
        }
        
        return EvaluationResult(
            component='gloss_to_text',
            metrics=metrics,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            num_samples=len(test_pairs),
            inference_time_ms=total_time / len(test_pairs)
        )


class SignRecognitionEvaluator:
    """Evaluate Sign Recognition model."""
    
    def __init__(self, model_path: str = None, vocab_path: str = None):
        import torch
        from train_recognition import SignRecognitionModel, BSLPoseDataset
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_path = model_path or str(project_root / "models/sign_recognition/best_model.pt")
        vocab_path = vocab_path or str(project_root / "models/sign_recognition/vocabulary.json")
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.gloss_to_idx = json.load(f)
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = SignRecognitionModel(
            input_dim=checkpoint.get('input_dim', 225),
            d_model=checkpoint.get('d_model', 256),
            num_classes=len(self.gloss_to_idx)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, data_dir: str = None, split: str = "test") -> EvaluationResult:
        """Evaluate on pose dataset."""
        import torch
        from torch.utils.data import DataLoader
        from train_recognition import BSLPoseDataset
        
        data_dir = data_dir or str(project_root / "data/poses")
        
        dataset = BSLPoseDataset(
            data_dir,
            split=split,
            gloss_to_idx=self.gloss_to_idx,
            augment=False,
            combine_splits=False
        )
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        y_true = []
        y_pred = []
        top5_preds = []
        total_time = 0
        
        with torch.no_grad():
            for batch in loader:
                poses = batch['poses'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['label']
                
                start = time.time()
                logits = self.model(poses, mask)
                total_time += (time.time() - start) * 1000
                
                # Top-1 predictions
                preds = logits.argmax(dim=1).cpu().numpy()
                
                # Top-5 predictions
                top5 = logits.topk(5, dim=1).indices.cpu().numpy()
                
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(preds.tolist())
                top5_preds.extend(top5.tolist())
        
        # Calculate metrics
        metrics = accuracy_metrics(y_true, y_pred, top5_preds)
        
        # Confusion matrix for top classes
        cm = confusion_matrix(y_true, y_pred)
        
        return EvaluationResult(
            component='sign_recognition',
            metrics=metrics,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            num_samples=len(y_true),
            inference_time_ms=total_time / len(y_true)
        )


# ============================================================
# Evaluation Report
# ============================================================

class EvaluationReport:
    """Generate evaluation report."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or project_root / "evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_result(self, result: EvaluationResult):
        """Add evaluation result."""
        self.results.append(result)
    
    def generate_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# BSL Translation System - Evaluation Report",
            f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n"
        ]
        
        for result in self.results:
            lines.append(f"## {result.component.replace('_', ' ').title()}")
            lines.append(f"\n- **Samples:** {result.num_samples}")
            lines.append(f"- **Avg Inference Time:** {result.inference_time_ms:.2f} ms")
            lines.append(f"\n### Metrics\n")
            
            for key, value in result.metrics.items():
                if key == 'per_class':
                    continue  # Skip detailed per-class metrics
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.4f}")
                else:
                    lines.append(f"- **{key}:** {value}")
            
            lines.append("\n---\n")
        
        report = '\n'.join(lines)
        
        # Save report
        report_path = self.output_dir / "evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save JSON
        json_path = self.output_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, title="Confusion Matrix"):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels)
        
        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        
        # Plot (top 20 classes only for readability)
        n = min(20, len(labels))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm[:n, :n], annot=True, fmt='.2f', 
                   xticklabels=labels[:n], yticklabels=labels[:n],
                   cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()


# ============================================================
# Test Data
# ============================================================

def get_test_data():
    """Load or create test data for evaluation."""
    
    # Text-to-Gloss test pairs
    text_to_gloss_tests = [
        ("Hello, my name is John", ["HELLO", "MY", "NAME", "JOHN"]),
        ("What time is the meeting?", ["WHAT", "TIME", "MEETING"]),
        ("Thank you very much", ["THANK", "YOU", "VERY", "MUCH"]),
        ("I don't understand", ["I", "NOT", "UNDERSTAND"]),
        ("Where is the bathroom?", ["WHERE", "BATHROOM"]),
        ("Nice to meet you", ["NICE", "MEET", "YOU"]),
        ("How are you today?", ["HOW", "YOU", "TODAY"]),
        ("Please help me", ["PLEASE", "HELP", "ME"]),
        ("I am learning sign language", ["I", "LEARN", "SIGN", "LANGUAGE"]),
        ("See you tomorrow", ["SEE", "YOU", "TOMORROW"]),
    ]
    
    # Gloss-to-Text test pairs
    gloss_to_text_tests = [
        (["HELLO", "MY", "NAME", "JOHN"], "Hello, my name is John."),
        (["WHAT", "TIME", "MEETING"], "What time is the meeting?"),
        (["THANK", "YOU", "MUCH"], "Thank you very much."),
        (["I", "NOT", "UNDERSTAND"], "I don't understand."),
        (["TOMORROW", "MEETING", "CANCEL"], "The meeting tomorrow is cancelled."),
    ]
    
    return {
        'text_to_gloss': text_to_gloss_tests,
        'gloss_to_text': gloss_to_text_tests
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate BSL Translation System")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--component", type=str, choices=[
        'text_to_gloss', 'gloss_to_text', 'recognition'
    ], help="Evaluate specific component")
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    report = EvaluationReport(args.output_dir)
    test_data = get_test_data()
    
    print("=" * 60)
    print("BSL TRANSLATION SYSTEM EVALUATION")
    print("=" * 60)
    
    if args.all or args.component == 'text_to_gloss':
        print("\n[1] Evaluating Text-to-Gloss...")
        try:
            evaluator = TextToGlossEvaluator()
            result = evaluator.evaluate(test_data['text_to_gloss'])
            report.add_result(result)
            print(f"    Accuracy: {result.metrics['avg_token_accuracy']:.4f}")
            print(f"    Coverage: {result.metrics['avg_coverage']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    if args.all or args.component == 'gloss_to_text':
        print("\n[2] Evaluating Gloss-to-Text...")
        try:
            evaluator = GlossToTextEvaluator(mode="groq")
            result = evaluator.evaluate(test_data['gloss_to_text'])
            report.add_result(result)
            print(f"    BLEU: {result.metrics['avg_bleu']:.4f}")
            print(f"    ROUGE-L: {result.metrics['avg_rouge_l']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    if args.all or args.component == 'recognition':
        print("\n[3] Evaluating Sign Recognition...")
        try:
            evaluator = SignRecognitionEvaluator()
            result = evaluator.evaluate(split="test")
            report.add_result(result)
            print(f"    Top-1 Accuracy: {result.metrics['accuracy']:.4f}")
            if result.metrics.get('top5_accuracy'):
                print(f"    Top-5 Accuracy: {result.metrics['top5_accuracy']:.4f}")
            print(f"    Macro F1: {result.metrics['macro_f1']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("Generating report...")
    report_text = report.generate_report()
    print(f"Report saved to: {report.output_dir}")
    
    print("\n" + report_text)


if __name__ == "__main__":
    main()
