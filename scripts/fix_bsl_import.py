"""Fix BSLDictRecognizer import to be global"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add global import after gradio import
old_import = '''try:
    import gradio as gr
except ImportError:
    print("Gradio required. Install with: pip install gradio")
    sys.exit(1)'''

new_import = '''try:
    import gradio as gr
except ImportError:
    print("Gradio required. Install with: pip install gradio")
    sys.exit(1)

# BSL Dict Recognizer (SWIN-based)
try:
    from src.inference.bsl_dict_recognizer import BSLDictRecognizer
    BSL_DICT_AVAILABLE = True
except ImportError:
    BSLDictRecognizer = None
    BSL_DICT_AVAILABLE = False
    print("Warning: BSLDictRecognizer not available")'''

content = content.replace(old_import, new_import)

# Update the get function to check availability
old_func = '''def get_bsl_dict_recognizer():
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

new_func = '''def get_bsl_dict_recognizer():
    """Lazy-load BSL dictionary recognizer (SWIN-based, 100% accuracy on 5203 signs)."""
    global _bsl_dict_recognizer
    if not BSL_DICT_AVAILABLE:
        print("BSL Dict Recognizer not available")
        return None
    if _bsl_dict_recognizer is None:
        try:
            import torch as th
            device = "cuda" if th.cuda.is_available() else "cpu"
            _bsl_dict_recognizer = BSLDictRecognizer(device=device)
            print(f"BSL Dict Recognizer loaded: {_bsl_dict_recognizer.glosses[:5]}...")
        except Exception as e:
            print(f"Failed to load BSL Dict Recognizer: {e}")
            return None
    return _bsl_dict_recognizer'''

content = content.replace(old_func, new_func)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixed global BSLDictRecognizer import")
