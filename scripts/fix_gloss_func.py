"""Fix function name in direction1_video_swin"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the function call
old_call = '''gloss_converter = get_gloss_to_text(mode, api_key)'''
new_call = '''gloss_converter = get_gloss_converter(mode, api_key)'''

content = content.replace(old_call, new_call)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixed get_gloss_converter call")
