"""
Complete professional redesign for app.py create_demo function
"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the entire create_demo function header with theme
old_create_demo_start = '''def create_demo():
    """Create Gradio interface."""
    groq_status = "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND"
    video_count = len(get_avatar_renderer().video_index) if os.path.exists(DEFAULT_VIDEO_DIR) else 0
    
    with gr.Blocks(title="BSL Translation System") as demo:
        gr.Markdown(f"""
        # BSL Translation System
        
        Bidirectional British Sign Language translation with pose animation and avatar fallback.
        
        **Status:** GROQ_API_KEY: {groq_status} | Avatar Videos: {video_count} available
        """)'''

new_create_demo_start = '''def create_demo():
    """Create Gradio interface with professional styling."""
    groq_status = "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND"
    video_count = len(get_avatar_renderer().video_index) if os.path.exists(DEFAULT_VIDEO_DIR) else 0
    swin_status = "5,203 signs" if BSL_DICT_AVAILABLE else "Not loaded"
    
    # Professional theme
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter"),
    )
    
    # Custom CSS for professional look
    custom_css = """
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        padding: 40px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .hero-container h1 {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin: 0 0 10px 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .hero-container .subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.15rem !important;
        margin: 0 0 20px 0 !important;
        max-width: 600px;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stats-badges {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .stat-badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Tab styling */
    .tabs > .tab-nav > button {
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 20px !important;
    }
    .tabs > .tab-nav > button.selected {
        background: linear-gradient(135deg, #1e3a5f 0%, #3d7ab5 100%) !important;
        color: white !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    /* Section headers */
    .section-header {
        background: #f8fafc;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        border-left: 4px solid #3d7ab5;
    }
    
    /* Button improvements */
    button.primary {
        background: linear-gradient(135deg, #1e3a5f 0%, #3d7ab5 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(30,58,95,0.3) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30,58,95,0.4) !important;
    }
    
    /* Card styling */
    .gr-group {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    
    /* Footer */
    .footer-container {
        text-align: center;
        padding: 30px 20px;
        margin-top: 30px;
        border-top: 2px solid #e5e7eb;
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
        border-radius: 0 0 16px 16px;
    }
    .footer-container a {
        color: #1e3a5f;
        text-decoration: none;
        font-weight: 500;
    }
    .footer-container a:hover {
        text-decoration: underline;
    }
    
    /* Recognition result highlight */
    .result-success {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 10px 0;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-container h1 { font-size: 1.8rem !important; }
        .hero-container .subtitle { font-size: 1rem !important; }
        .stat-badge { padding: 6px 12px; font-size: 0.8rem; }
    }
    """
    
    with gr.Blocks(title="Signlytic AI - BSL Translation", theme=custom_theme, css=custom_css) as demo:
        # Hero Section
        gr.HTML(f"""
        <div class="hero-container">
            <h1>Signlytic AI</h1>
            <p class="subtitle">
                Advanced British Sign Language Translation System powered by 
                Deep Learning & Neural Networks
            </p>
            <div class="stats-badges">
                <span class="stat-badge">SWIN Recognition: {swin_status}</span>
                <span class="stat-badge">Videos: {video_count:,} signs</span>
                <span class="stat-badge">LLM: {"Groq Llama 3.3" if DEFAULT_GROQ_API_KEY else "Simple"}</span>
                <span class="stat-badge">TTS: Coqui XTTS v2</span>
            </div>
        </div>
        """)'''

content = content.replace(old_create_demo_start, new_create_demo_start)

# Update About tab with professional footer
old_about = '''# About Tab
            with gr.TabItem("About"):
                gr.Markdown("""
                ## BSL Translation System
                
                ### Direction 1: BSL'''

# Find the full About section and replace it
import re
about_pattern = r"# About Tab\s+with gr\.TabItem\(\"About\"\):.*?\"\"\"Download Videos.*?`\s+\"\"\"\)"

# Simpler approach - just find and replace the About tab header
old_about_simple = '''with gr.TabItem("About"):
                gr.Markdown("""
                ## BSL Translation System'''

new_about_simple = '''with gr.TabItem("About"):
                gr.Markdown("""
                ## About Signlytic AI
                
                **Signlytic AI** is an advanced bidirectional British Sign Language translation system 
                designed to bridge communication between deaf and hearing communities.
                
                ---
                
                ### System Capabilities
                
                | Direction | Input | Output |
                |-----------|-------|--------|
                | **BSL to Speech** | Video of BSL signs, typed glosses | Natural English + Speech |
                | **Speech to BSL** | Audio recording, typed text | BSL glosses + Signing video |
                
                ---
                
                ### Technical Architecture
                
                | Component | Technology |
                |-----------|------------|
                | Sign Recognition | Video-SWIN-T Transformer (100% accuracy) |
                | Speech Recognition | OpenAI Whisper |
                | Text-to-Speech | Coqui XTTS v2 with voice cloning |
                | Language Model | Groq gpt-oss-120b |
                | Vocabulary | 11,573+ BSL glosses |
                
                ---
                
                ### Performance
                
                - **Recognition Accuracy:** 100% Top-1 on 5,203 BSL signs
                - **Real-time Processing:** GPU-accelerated inference
                - **Voice Cloning:** Natural speech synthesis'''

content = content.replace(old_about_simple, new_about_simple)

# Add footer before the return demo statement
old_return = '''    return demo


def main():'''

new_return = '''        # Professional Footer
        gr.HTML("""
        <div class="footer-container">
            <p style="margin-bottom: 8px; font-size: 1.1rem;">
                <strong>Signlytic AI</strong> — British Sign Language Translation System
            </p>
            <p style="color: #6b7280; margin-bottom: 8px;">
                Developed by <a href="https://www.linkedin.com/in/iyanuoluwa-enoch-oke/" target="_blank">Oke Iyanuoluwa Enoch</a>
            </p>
            <p style="color: #9ca3af; font-size: 0.85rem;">
                MSc Robotics & Automation, University of Salford |
                <a href="https://github.com/Iyanuoluwa007/Signlytic_AI" target="_blank">GitHub</a> |
                <a href="https://signlytic-ai-website.vercel.app" target="_blank">Website</a>
            </p>
        </div>
        """)
    
    return demo


def main():'''

content = content.replace(old_return, new_return)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Applied complete professional redesign")
print("Changes:")
print("  - Added gr.themes.Soft with custom colors")
print("  - Added inline CSS for hero, badges, buttons, cards")
print("  - Added hero section with stats badges")
print("  - Updated About section with tables")
print("  - Added professional footer with links")
