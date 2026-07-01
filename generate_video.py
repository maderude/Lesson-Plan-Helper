import os
import argparse
from openai import OpenAI
from moviepy.editor import VideoFileClip, AudioFileClip

# The exact script from our Kaggle write-up
VOICEOVER_TEXT = """
Hello! Welcome to my Kaggle Capstone project: the Lesson Plan Helper. 

Teachers spend an average of ten to fifteen hours every week outside of school just planning lessons. It's a high-stakes, repetitive task that requires balancing strict state standards, realistic pacing, and diverse classroom accommodations. Most single-prompt AI tools fail at this because they hallucinate standards or drop constraints. Our project solves this.

To solve this, we didn't just use a massive prompt; we built a stateful, multi-agent workflow using Python, Streamlit, and LangGraph. 

Why agents? Because lesson planning is a multi-constraint problem. By separating the work into specialized creator, evaluator, and reviser roles, the system self-corrects. A Planning Agent drafts the lesson, a Review Agent grades it, and a Rewrite Agent fixes only the failing sections before the user ever sees it.

Let's look at a demo. We built a Human-in-the-Loop architecture because education requires trust. 

When a teacher enters their topic and time constraints, the system first retrieves real state standards. The workflow actually pauses here. The teacher must review and confirm the exact standard they want to target before any drafting begins. This grounds the AI entirely in classroom reality.

Once approved, the Multi-Agent graph takes over. 

While it drafts, I want to highlight our Model Context Protocol, or MCP server. We built a local MCP server that exposes strict rubric scoring and validation tools. Instead of letting the Review Agent vaguely critique the draft, the MCP server programmatically checks if the pacing adds up to exactly 45 minutes, and if the assessment actually matches the cognitive verb of the standard.

Here is the final output. If the initial draft failed our MCP rubric—say, it planned 60 minutes of activities for a 45-minute slot—the Rewrite Agent already looped in to fix it. 

The teacher is presented with a classroom-ready, fully aligned lesson broken into actionable cards. We also built robust parser fallbacks in the UI to ensure that even if the LLM strips markdown formatting during rewrites, the UI always renders perfectly.

Teachers can then seamlessly export this to a formatted PDF or Word Document to hand in to their administration. 

Building this with Antigravity and the agent development kit allowed us to rapidly iterate on the complex orchestration and parser logic required to make this robust.

Lesson Plan Helper turns a 45-minute planning burden into a 2-minute fast-review workflow. It's a true Agent for Good. Thank you for watching!
"""

def generate_audio(output_audio_path="voiceover.mp3"):
    print("🎙️ Generating audio via OpenAI TTS...")
    client = OpenAI() # Uses OPENAI_API_KEY from environment
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx", # 'onyx' is a deep, professional narrator voice. Alternatives: 'alloy', 'echo', 'fable', 'nova', 'shimmer'
        input=VOICEOVER_TEXT
    )
    
    response.stream_to_file(output_audio_path)
    print(f"✅ Audio saved to {output_audio_path}")
    return output_audio_path

def combine_audio_video(video_path, audio_path, output_path="final_submission.mp4"):
    print(f"🎬 Combining {video_path} and {audio_path}...")
    
    # Load the silent video and the generated audio
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    # If the video is longer than the audio, it will cut off. 
    # If the audio is longer, we can freeze the last frame of the video or let it go black.
    # We will set the video's audio track to the new audio clip.
    final_video = video_clip.set_audio(audio_clip)
    
    # Write the result to a file
    print("⏳ Writing final video file (this may take a minute)...")
    final_video.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        fps=24,
        preset="ultrafast"
    )
    print(f"🎉 Final video successfully saved as {output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle submission video with AI voiceover.")
    parser.add_argument("--video", type=str, required=True, help="Path to your silent screen recording (e.g. demo.mp4)")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Could not find video file at '{args.video}'")
        exit(1)
        
    audio_file = generate_audio()
    combine_audio_video(args.video, audio_file)
