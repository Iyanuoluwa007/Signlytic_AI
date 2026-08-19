"""
Patch to add professional styling to app.py
"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add custom CSS constant after imports
CUSTOM_CSS = '''
# =============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# =============================================================================
CUSTOM_CSS = """
/* Main container */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

/* Hero styling */
.hero-section {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
}

.hero-section h1 {
    color: white !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.hero-section p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.15rem !important;
    max-width: 700px;
    margin: 0 auto 1rem auto !important;
}

/* Stats badges */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.stat-badge {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    color: white;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Tab improvements */
.tabs {
    border-radius: 12px !important;
    overflow: hidden;
}

button.selected {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%) !important;
    color: white !important;
}

/* Card sections */
.gr-box {
    border-radius: 12px !important;
    border: 1px solid #e0e0e0 !important;
}

/* Primary buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #1e3a5f 0%, #3d7ab5 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(30,58,95,0.3) !important;
}

/* Secondary buttons */
.gr-button-secondary {
    border: 2px solid #1e3a5f !important;
    color: #1e3a5f !important;
    font-weight: 500 !important;
}

/* Recognition highlight */
.recognition-success {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Footer */
.footer-section {
    text-align: center;
    padding: 2rem;
    margin-top: 2rem;
    border-top: 2px solid #e5e7eb;
    background: #f9fafb;
    border-radius: 0 0 12px 12px;
}

.footer-section a {
    color: #1e3a5f;
    text-decoration: none;
    font-weight: 500;
}

.footer-section a:hover {
    text-decoration: underline;
}

/* Feature icons */
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-section h1 { font-size: 1.8rem !important; }
    .hero-section p { font-size: 1rem !important; }
    .stat-badge { padding: 0.4rem 0.8rem; font-size: 0.85rem; }
}
"""

'''

# Find the location to insert CSS (after DEFAULT_GROQ_API_KEY)
old_groq_line = 'DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()'
new_groq_line = old_groq_line + CUSTOM_CSS

content = content.replace(old_groq_line, new_groq_line)

# 2. Update create_demo function with professional theme and hero
old_create_demo = '''def create_demo():
    """Create Gradio interface."""
    groq_status = "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND"
    video_count = len(get_avatar_renderer().video_index) if os.path.exists(DEFAULT_VIDEO_DIR) else 0
    
    with gr.Blocks(title="BSL Translation System") as demo:
        gr.Markdown(f"""
        # BSL Translation System
        
        Bidirectional British Sign Language translation with pose animation and avatar fallback.
        
        **Status:** GROQ_API_KEY: {groq_status} | Avatar Videos: {video_count} available
        """)'''

new_create_demo = '''def create_demo():
    """Create Gradio interface with professional styling."""
    groq_status = "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND"
    video_count = len(get_avatar_renderer().video_index) if os.path.exists(DEFAULT_VIDEO_DIR) else 0
    swin_status = "5,203 signs" if BSL_DICT_AVAILABLE else "Not loaded"
    
    # Use Soft theme for professional look
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill="linear-gradient(135deg, #1e3a5f 0%, #3d7ab5 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #2d5a87 0%, #4d8ac5 100%)",
        block_title_text_weight="600",
        block_label_text_weight="500",
    )
    
    with gr.Blocks(title="Signlytic AI - BSL Translation", theme=theme, css=CUSTOM_CSS) as demo:
        # Hero Section
        gr.HTML(f"""
        <div class="hero-section">
            <h1>Signlytic AI</h1>
            <p>
                Advanced British Sign Language Translation System powered by 
                Video-SWIN Transformers, Whisper ASR, and Neural TTS
            </p>
            <div class="stats-row">
                <span class="stat-badge">SWIN Recognition: {swin_status}</span>
                <span class="stat-badge">Avatar Videos: {video_count:,}</span>
                <span class="stat-badge">LLM: {'Groq gpt-oss-120b' if DEFAULT_GROQ_API_KEY else 'Simple Mode'}</span>
                <span class="stat-badge">TTS: Coqui XTTS v2</span>
            </div>
        </div>
        """)'''

content = content.replace(old_create_demo, new_create_demo)

# 3. Update Direction 1 tab header
old_d1_header = '''with gr.TabItem("Direction 1: BSL to Speech"):
                gr.Markdown("""
                ### BSL Glosses → Natural English → Speech
                
                Choose one input style below for a cleaner workflow.
                - Option A: Type glosses
                - Option B: Camera/upload video (record then recognize)
                - Option C: Live realtime camera in-app preview (no popup window)
                """)'''

new_d1_header = '''with gr.TabItem("BSL → Speech", id="bsl-to-speech"):
                gr.Markdown("""
                ## Recognize BSL Signs & Convert to Speech
                
                Upload a BSL video or type glosses to get natural English text and speech output.
                
                **Recommended:** Use the **SWIN Recognition** button for best accuracy (5,203 signs supported).
                """)'''

content = content.replace(old_d1_header, new_d1_header)

# 4. Update Direction 2 tab header  
old_d2_header = '''with gr.TabItem("Direction 2: Speech to BSL"):
                gr.Markdown("""
                ### Speech/Text -> BSL Glosses -> Animated Signing

                Choose one input style below, then pick a render engine.
                - Pose Animator (default): in-app 2D hand-sign animation + MP4 output
                - Legacy Clip Avatar: existing sign video concatenation flow
                """)'''

new_d2_header = '''with gr.TabItem("Speech → BSL", id="speech-to-bsl"):
                gr.Markdown("""
                ## Convert Speech or Text to BSL Signing
                
                Record audio or type text to generate BSL glosses and animated signing videos.
                """)'''

content = content.replace(old_d2_header, new_d2_header)

# 5. Update About tab with professional footer
old_about = '''# About Tab
            with gr.TabItem("About"):
                gr.Markdown("""
                ## BSL Translation System
                
                ### Direction 1: BSL → Speech
                - Input: BSL glosses, camera recording/upload, or live realtime camera
                - Output: Natural English text + synthesized speech (Coqui TTS)
                
                ### Direction 2: Speech → BSL
                - Input: Speech via record/upload selector, or text
                - Output: BSL glosses + Pose animation preview + MP4 signing video
                
                ### Technical Stack
                - **ASR:** OpenAI Whisper
                - **TTS:** Coqui XTTS v2 with voice cloning
                - **Gloss-to-Text:** Groq gpt-oss-120b
                - **Signing Renderer:** 2D pose animator (default) + legacy clip avatar fallback
                - **Vocabulary:** 11,573 BSL glosses
                
                ### Download Videos
                To enable avatar rendering, download BSL sign videos:
`
                python scripts/download_bsl_videos.py --limit 500
`
                """)'''

new_about = '''# About Tab
            with gr.TabItem("About", id="about"):
                gr.Markdown("""
                ## About Signlytic AI
                
                **Signlytic AI** is an advanced bidirectional British Sign Language translation system 
                designed to bridge communication between deaf and hearing communities.
                
                ---
                
                ### Features
                
                | Direction | Input | Output |
                |-----------|-------|--------|
                | **BSL → Speech** | Video of BSL signs, typed glosses | Natural English text + synthesized speech |
                | **Speech → BSL** | Audio recording, typed text | BSL glosses + animated signing video |
                
                ---
                
                ### Technical Architecture
                
                | Component | Technology |
                |-----------|------------|
                | **Sign Recognition** | Video-SWIN-T Transformer (100% accuracy on 5,203 signs) |
                | **Speech Recognition** | OpenAI Whisper (base model) |
                | **Text-to-Speech** | Coqui XTTS v2 with voice cloning |
                | **Language Model** | Groq gpt-oss-120b for gloss↔text conversion |
                | **Avatar Rendering** | 2D Pose Animator + Video Concatenation |
                | **Vocabulary** | 11,573+ BSL glosses |
                
                ---
                
                ### Performance Metrics
                
                - **SWIN Recognition Accuracy:** 100% Top-1 on BSL Dictionary videos
                - **Supported Signs:** 5,203 unique BSL signs
                - **Real-time Processing:** GPU-accelerated inference
                
                ---
                
                ### Getting Started
`ash
                # Download BSL sign videos for avatar rendering
                python scripts/download_bsl_videos.py --limit 500
                
                # Run the application
                python app.py --share  # Creates public link
`
                """)
                
                # Footer
                gr.HTML("""
                <div class="footer-section">
                    <p style="margin-bottom: 0.5rem;">
                        <strong>Signlytic AI</strong> — Developed by 
                        <a href="https://www.linkedin.com/in/iyanuoluwa-enoch-oke/" target="_blank">Oke Iyanuoluwa Enoch</a>
                    </p>
                    <p style="color: #6b7280; font-size: 0.9rem;">
                        MSc Robotics & Automation, University of Salford | 
                        <a href="https://github.com/Iyanuoluwa007/Signlytic_AI" target="_blank">GitHub</a> | 
                        <a href="https://signlytic-ai-website.vercel.app" target="_blank">Website</a>
                    </p>
                </div>
                """)'''

content = content.replace(old_about, new_about)

# Save the updated app.py
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Applied professional styling to app.py")
print("Changes made:")
print("  - Added custom CSS for hero section, badges, buttons")
print("  - Applied gr.themes.Soft with custom colors")
print("  - Updated hero section with stats badges")
print("  - Improved tab headers with clearer descriptions")
print("  - Added professional About section with tables")
print("  - Added footer with author attribution")
