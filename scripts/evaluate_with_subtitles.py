"""
BSL Recognition Evaluation with Visual Results
Tests multiple videos and creates a report with predictions.
"""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from pathlib import Path
from src.inference.bsl_dict_recognizer import BSLDictRecognizer
import json
from datetime import datetime

def create_evaluation_video():
    """Create a video showing BSL recognition results with subtitles."""
    
    print("="*70)
    print("BSL RECOGNITION EVALUATION")
    print("="*70)
    
    # Initialize recognizer
    recognizer = BSLDictRecognizer()
    
    # Test videos - common words
    test_words = [
        'hello', 'help', 'good', 'bad', 'yes', 'no', 'love', 'work', 
        'family', 'friend', 'eat', 'drink', 'morning', 'afternoon',
        'evening', 'night', 'today', 'tomorrow', 'name', 'sad'
    ]
    
    video_dir = Path("data/videos/bsl_signs")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    # Process each video
    print("\nTesting videos...")
    print("-" * 70)
    
    for word in test_words:
        video_path = video_dir / f"{word}.mp4"
        if not video_path.exists():
            continue
        
        # Recognize
        preds = recognizer.recognize(str(video_path), top_k=5)
        
        pred_top1 = preds[0][0]
        conf_top1 = preds[0][1] * 100
        top3 = [p[0] for p in preds[:3]]
        
        is_top1 = pred_top1 == word
        is_top3 = word in top3
        
        correct_top1 += int(is_top1)
        correct_top3 += int(is_top3)
        total += 1
        
        status = "[OK]" if is_top1 else ("[TOP3]" if is_top3 else "[MISS]")
        
        result = {
            'expected': word,
            'predicted': pred_top1,
            'confidence': conf_top1,
            'top3': top3,
            'correct_top1': is_top1,
            'correct_top3': is_top3,
        }
        results.append(result)
        
        print(f"{status} Expected: {word:15} | Predicted: {pred_top1:15} ({conf_top1:5.1f}%)")
    
    # Summary
    print("-" * 70)
    print(f"\nRESULTS SUMMARY:")
    print(f"  Total tested: {total}")
    print(f"  Top-1 Accuracy: {correct_top1}/{total} = {correct_top1/total*100:.1f}%")
    print(f"  Top-3 Accuracy: {correct_top3}/{total} = {correct_top3/total*100:.1f}%")
    
    # Save JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': total,
        'top1_accuracy': correct_top1 / total * 100,
        'top3_accuracy': correct_top3 / total * 100,
        'results': results
    }
    
    report_path = output_dir / "bsl_evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
    
    # Create combined video with subtitles
    print("\nCreating evaluation video with subtitles...")
    create_subtitle_video(results, video_dir, output_dir)
    
    return results


def create_subtitle_video(results, video_dir, output_dir):
    """Create a video showing each sign with expected/predicted subtitles."""
    
    output_path = output_dir / "bsl_evaluation_video.mp4"
    
    # Video settings
    frame_width = 640
    frame_height = 480
    fps = 25
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    for result in results:
        word = result['expected']
        video_path = video_dir / f"{word}.mp4"
        
        if not video_path.exists():
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Add subtitle background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 100), (frame_width, frame_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Add expected label (green)
            expected_text = f"Expected: {result['expected'].upper()}"
            cv2.putText(frame, expected_text, (20, frame_height - 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add predicted label (green if correct, red if wrong)
            color = (0, 255, 0) if result['correct_top1'] else (0, 0, 255)
            pred_text = f"Predicted: {result['predicted'].upper()} ({result['confidence']:.0f}%)"
            cv2.putText(frame, pred_text, (20, frame_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add status indicator
            status = "CORRECT" if result['correct_top1'] else "WRONG"
            status_color = (0, 255, 0) if result['correct_top1'] else (0, 0, 255)
            cv2.putText(frame, status, (frame_width - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        
        # Add blank frames between clips
        blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for _ in range(10):
            out.write(blank)
    
    out.release()
    print(f"Evaluation video saved: {output_path}")


if __name__ == "__main__":
    create_evaluation_video()
