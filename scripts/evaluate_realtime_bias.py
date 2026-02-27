"""
Evaluate realtime recognition collapse metrics (especially ABOUT false positives).
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_recognition import SignRecognitionModel, BSLPoseDataset


class RealtimeBiasEvaluator:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        class_stats_path: Optional[str] = None,
        device: Optional[str] = None,
        abstain_threshold: float = 0.12,
        margin_threshold: float = 0.02,
        logit_adjustment_tau: float = 0.7,
        disable_logit_adjustment: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.abstain_threshold = abstain_threshold
        self.margin_threshold = margin_threshold
        self.logit_adjustment_tau = logit_adjustment_tau
        self.disable_logit_adjustment = disable_logit_adjustment

        with open(vocab_path, "r") as f:
            self.gloss_to_idx = json.load(f)
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
        self.about_idx = self.gloss_to_idx.get("ABOUT")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = SignRecognitionModel(
            input_dim=ckpt.get("input_dim", 225),
            d_model=ckpt.get("d_model", 256),
            nhead=ckpt.get("nhead", 8),
            num_layers=ckpt.get("num_layers", 4),
            dim_feedforward=ckpt.get("dim_feedforward", ckpt.get("d_model", 256) * 4),
            num_classes=len(self.gloss_to_idx),
            dropout=ckpt.get("dropout", 0.1),
            max_frames=ckpt.get("max_frames", 64),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        priors = ckpt.get("class_priors")
        if priors is None and class_stats_path and Path(class_stats_path).exists():
            with open(class_stats_path, "r") as f:
                stats = json.load(f)
            p = np.zeros(len(self.gloss_to_idx), dtype=np.float32)
            for gloss, idx in self.gloss_to_idx.items():
                p[idx] = float(stats.get("class_priors", {}).get(gloss, 0.0))
            priors = p.tolist()
        if priors is not None:
            p = np.array(priors, dtype=np.float32)
            p = np.clip(p, 1e-8, None)
            p = p / p.sum()
            self.class_priors = torch.tensor(p, dtype=torch.float32, device=self.device)
        else:
            self.class_priors = None

    def _apply_postprocess(self, logits: torch.Tensor) -> torch.Tensor:
        if self.disable_logit_adjustment:
            return logits
        if self.class_priors is None:
            return logits
        return logits - self.logit_adjustment_tau * torch.log(self.class_priors.unsqueeze(0))

    def _abstain(self, probs: torch.Tensor) -> torch.Tensor:
        top2 = probs.topk(k=2, dim=1)
        top1_conf = top2.values[:, 0]
        top2_conf = top2.values[:, 1]
        margin = top1_conf - top2_conf
        return (top1_conf < self.abstain_threshold) | (margin < self.margin_threshold)

    @torch.no_grad()
    def evaluate_dataset(self, dataset: BSLPoseDataset, batch_size: int = 32, max_samples: Optional[int] = None) -> Dict:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total = 0
        top1_correct = 0
        top5_correct = 0
        abstain_count = 0
        about_pred_count = 0
        about_true_count = 0
        about_fp = 0
        about_fp_sources = Counter()

        for batch in loader:
            poses = batch["poses"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            valid_rows = mask.any(dim=1)
            if not bool(valid_rows.any()):
                continue

            logits = self.model(poses, mask)
            logits = self._apply_postprocess(logits)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            top5 = probs.topk(k=5, dim=1).indices
            abstain = self._abstain(probs)

            for i in range(labels.size(0)):
                if max_samples is not None and total >= max_samples:
                    break
                total += 1
                label = int(labels[i].item())
                pred = int(preds[i].item())
                if abstain[i].item():
                    abstain_count += 1
                    continue

                if pred == label:
                    top1_correct += 1
                if label in top5[i].tolist():
                    top5_correct += 1

                if self.about_idx is not None:
                    if pred == self.about_idx:
                        about_pred_count += 1
                    if label == self.about_idx:
                        about_true_count += 1
                    if pred == self.about_idx and label != self.about_idx:
                        about_fp += 1
                        about_fp_sources[self.idx_to_gloss.get(label, "UNKNOWN")] += 1

            if max_samples is not None and total >= max_samples:
                break

        accepted = total - abstain_count
        non_abstained = max(accepted, 1)
        report = {
            "num_samples": total,
            "abstain_count": abstain_count,
            "abstain_rate": abstain_count / max(total, 1),
            "accepted_count": max(accepted, 0),
            "top1_accuracy_overall": top1_correct / max(total, 1),
            "top5_accuracy_overall": top5_correct / max(total, 1),
            "top1_accuracy_non_abstain": top1_correct / non_abstained,
            "top5_accuracy_non_abstain": top5_correct / non_abstained,
            "about_prediction_rate_non_abstain": about_pred_count / non_abstained,
            "about_true_rate": about_true_count / max(total, 1),
            "about_fp_count": about_fp,
            "about_fp_share_non_abstain": about_fp / non_abstained,
            "about_fp_sources": dict(about_fp_sources.most_common(50)),
        }
        return report

    @torch.no_grad()
    def evaluate_static_abstain(self, num_windows: int = 200, frames: int = 64, feat_dim: int = 225) -> Dict:
        zeros = torch.zeros((num_windows, frames, feat_dim), dtype=torch.float32, device=self.device)
        noise = torch.randn((num_windows, frames, feat_dim), dtype=torch.float32, device=self.device) * 1e-4
        poses = torch.cat([zeros, noise], dim=0)
        mask = torch.zeros((poses.size(0), frames), dtype=torch.bool, device=self.device)
        mask[:, 0] = True

        logits = self.model(poses, mask)
        logits = self._apply_postprocess(logits)
        probs = F.softmax(logits, dim=1)
        abstain = self._abstain(probs)
        abstain_rate = float(abstain.float().mean().item())
        return {
            "num_windows": int(poses.size(0)),
            "abstain_rate": abstain_rate,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate realtime ABOUT collapse bias")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--vocab", type=str, default=None, help="Vocabulary JSON")
    parser.add_argument("--class-stats", type=str, default=None, help="Class stats JSON")
    parser.add_argument("--data-dir", type=str, default=None, help="Pose data directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--abstain-threshold", type=float, default=0.12)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.7)
    parser.add_argument("--disable-logit-adjustment", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Output report JSON")
    args = parser.parse_args()

    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    model_path = Path(args.model) if args.model else project_root / "models" / "sign_recognition" / "best_model.pt"
    vocab_path = Path(args.vocab) if args.vocab else project_root / "models" / "sign_recognition" / "vocabulary.json"
    class_stats = Path(args.class_stats) if args.class_stats else project_root / "models" / "sign_recognition" / "class_stats.json"
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "poses"
    output_path = Path(args.output) if args.output else project_root / "evaluation" / "realtime_bias_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = RealtimeBiasEvaluator(
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        class_stats_path=str(class_stats) if class_stats.exists() else None,
        abstain_threshold=args.abstain_threshold,
        margin_threshold=args.margin_threshold,
        logit_adjustment_tau=args.logit_adjustment_tau,
        disable_logit_adjustment=args.disable_logit_adjustment,
    )

    dataset = BSLPoseDataset(
        data_dir=str(data_dir),
        split=args.split,
        gloss_to_idx=evaluator.gloss_to_idx,
        augment=False,
        combine_splits=False,
        normalize_poses=True,
    )
    data_metrics = evaluator.evaluate_dataset(dataset, batch_size=args.batch_size, max_samples=args.max_samples)
    static_metrics = evaluator.evaluate_static_abstain()

    report = {
        "model": str(model_path),
        "split": args.split,
        "abstain_threshold": args.abstain_threshold,
        "margin_threshold": args.margin_threshold,
        "logit_adjustment_tau": args.logit_adjustment_tau,
        "disable_logit_adjustment": args.disable_logit_adjustment,
        "dataset_metrics": data_metrics,
        "static_no_sign_metrics": static_metrics,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
