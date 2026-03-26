"""Fix torch import in get_bsl_dict_recognizer"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the function to import torch
old_func = '''def get_bsl_dict_recognizer():
    """Lazy-load BSL dictionary recognizer (SWIN-based, 100% accuracy on 5203 signs)."""
    global _bsl_dict_recognizer
    if _bsl_dict_recognizer is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _bsl_dict_recognizer = BSLDictRecognizer(device=device)
        except Exception as e:
            print(f"Failed to load BSL Dict Recognizer: {e}")
            return None
    return _bsl_dict_recognizer'''

new_func = '''def get_bsl_dict_recognizer():
    """Lazy-load BSL dictionary recognizer (SWIN-based, 100% accuracy on 5203 signs)."""
    global _bsl_dict_recognizer
    if _bsl_dict_recognizer is None:
        try:
            import torch as th
            device = "cuda" if th.cuda.is_available() else "cpu"
            _bsl_dict_recognizer = BSLDictRecognizer(device=device)
        except Exception as e:
            print(f"Failed to load BSL Dict Recognizer: {e}")
            return None
    return _bsl_dict_recognizer'''

content = content.replace(old_func, new_func)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixed torch import")
