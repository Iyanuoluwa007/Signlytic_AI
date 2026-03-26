"""Direct fix to apply theme and CSS to gr.Blocks"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Replace the gr.Blocks line to include theme and css
old_blocks = 'with gr.Blocks(title="BSL Translation System") as demo:'
new_blocks = '''with gr.Blocks(
        title="Signlytic AI - BSL Translation",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate", 
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=CUSTOM_CSS
    ) as demo:'''

content = content.replace(old_blocks, new_blocks)

# Fix 2: Replace the old markdown header with HTML hero
old_header = '''gr.Markdown(f"""
        # BSL Translation System
        
        Bidirectional British Sign Language translation with pose animation and avatar fallback.
        
        **Status:** GROQ_API_KEY: {groq_status} | Avatar Videos: {video_count} available
        """)'''

new_header = '''# Hero Section
        swin_status = "5,203 signs" if BSL_DICT_AVAILABLE else "Not loaded"
        gr.HTML(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%); 
                    padding: 40px 30px; border-radius: 16px; margin-bottom: 24px; 
                    text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.15);">
            <h1 style="color: white; font-size: 2.5rem; font-weight: 700; margin: 0 0 10px 0; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                Signlytic AI
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0 0 20px 0;">
                Advanced British Sign Language Translation System
            </p>
            <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 20px; 
                            color: white; font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.2);">
                    SWIN: {swin_status}
                </span>
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 20px; 
                            color: white; font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.2);">
                    Videos: {video_count:,}
                </span>
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 20px; 
                            color: white; font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.2);">
                    LLM: {"Groq" if groq_status == "FOUND" else "Simple"}
                </span>
            </div>
        </div>
        """)'''

content = content.replace(old_header, new_header)

# Fix 3: Update footer text
content = content.replace(
    "MSc Robotics & Automation, University of Salford",
    "Independent Robotics & AI Systems Engineer"
)

# Fix 4: Update launch to include pwa=True
old_launch = '''demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )'''

new_launch = '''demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        pwa=True
    )'''

content = content.replace(old_launch, new_launch)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Applied direct fixes:")
print("  - Added theme=gr.themes.Soft() to gr.Blocks")
print("  - Added css=CUSTOM_CSS to gr.Blocks")
print("  - Replaced Markdown header with HTML hero")
print("  - Updated footer attribution")
print("  - Added pwa=True to launch()")
