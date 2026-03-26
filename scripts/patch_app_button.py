"""
Patch to add SWIN recognition button to Gradio interface
"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add SWIN button after the existing video button
old_btn = '''d1_video_btn = gr.Button("Recognize from Camera/Video", variant="secondary")'''

new_btn = '''d1_video_btn = gr.Button("Recognize (Pose-based)", variant="secondary")
                        d1_swin_btn = gr.Button("Recognize (SWIN - 5203 signs)", variant="primary")'''

content = content.replace(old_btn, new_btn)

# Add click handler for SWIN button after the existing video button handler
old_handler = '''d1_video_btn.click(
                    fn=direction1_video_to_speech,
                    inputs=[d1_video_input, d1_mode, d1_api_key],
                    outputs=[d1_video_glosses, d1_text_output, d1_audio_output]
                )'''

new_handler = '''d1_video_btn.click(
                    fn=direction1_video_to_speech,
                    inputs=[d1_video_input, d1_mode, d1_api_key],
                    outputs=[d1_video_glosses, d1_text_output, d1_audio_output]
                )
                d1_swin_btn.click(
                    fn=direction1_video_swin,
                    inputs=[d1_video_input, d1_mode, d1_api_key],
                    outputs=[d1_video_glosses, d1_text_output, d1_audio_output]
                )'''

content = content.replace(old_handler, new_handler)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Added SWIN button to Gradio interface")
