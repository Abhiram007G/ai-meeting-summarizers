import os
import json
import tempfile
from pathlib import Path
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment

# Check for OpenAI API key in Streamlit secrets
if 'OPENAI_API_KEY' not in st.secrets:
    st.error('Error: OPENAI_API_KEY not found in Streamlit secrets. Please add it to your secrets.')
    st.markdown("""
    ### How to add your OpenAI API key:
    1. Local Development:
        - Create a `.streamlit/secrets.toml` file
        - Add: `OPENAI_API_KEY = "your-api-key"`
    2. Streamlit Cloud:
        - Go to your app settings
        - Add your API key in the secrets section
    """)
    st.stop()

# Initialize OpenAI client with API key from secrets
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Define the JSON file to store transcriptions
JSON_FILE = "transcriptions.json"

def read_transcriptions():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

def write_transcriptions(data):
    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def split_audio(audio_file, chunk_duration_ms=600000):
    """Split audio file into chunks of specified duration."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    try:
        audio = AudioSegment.from_file(temp_audio_path)
        duration = len(audio)
        chunks = []
        
        # Calculate number of chunks
        num_chunks = (duration + chunk_duration_ms - 1) // chunk_duration_ms
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Audio duration: {duration/1000:.2f} seconds")
        status_text.text(f"Splitting into {num_chunks} chunks...")
        
        for i in range(num_chunks):
            start_time = i * chunk_duration_ms
            end_time = min((i + 1) * chunk_duration_ms, duration)
            chunk = audio[start_time:end_time]
            chunks.append(chunk)
            
            # Update progress
            progress = (i + 1) / num_chunks
            progress_bar.progress(progress)
            status_text.text(f"Created chunk {i+1}/{num_chunks} ({len(chunk)/1000:.2f} seconds)")
        
        return chunks
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def transcribe_chunk(chunk, chunk_index):
    """Transcribe a single audio chunk."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = temp_file.name
        try:
            chunk.export(temp_path, format="mp3")
            
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def summarize_transcript(transcript):
    """Generate a summary of the transcript using GPT-4."""
    try:
        with st.spinner("Generating summary using GPT-4..."):
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Please provide a comprehensive summary of this transcript in clear, detailed bullet points. Each bullet point should be 2-3 lines long and capture the complete context and key learnings. Format as follows:

Key Learnings and Insights:
• [Each bullet should provide complete context and key takeaway in 2-3 lines]
• [Focus on actionable insights, important concepts, and practical applications]
• [Include specific examples or case studies mentioned]

Main Discussion Points:
• [Capture major topics with their context and significance in 2-3 lines]
• [Include any important statistics, research findings, or evidence presented]
• [Note any methodologies, techniques, or approaches discussed]

Action Items and Applications:
• [List practical applications and implementation steps in 2-3 lines]
• [Include any recommended practices or suggested approaches]
• [Note any tools, resources, or references mentioned]

Remember to:
- Make each bullet point self-contained and comprehensive
- Include specific details while maintaining clarity
- Preserve all key information and learnings
- Use clear, professional language

Transcript:
{transcript}"""
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def process_audio_file(audio_file):
    """Process an audio file by splitting it into chunks and transcribing each chunk."""
    
    # Create placeholders for status updates
    status = st.empty()
    progress_bar = st.progress(0)
    
    # Split audio into chunks
    status.text("Splitting audio into chunks...")
    chunks = split_audio(audio_file)
    
    # Transcribe each chunk
    transcriptions = []
    for i, chunk in enumerate(chunks):
        status.text(f"Transcribing chunk {i+1}/{len(chunks)}...")
        progress_bar.progress((i + 1) / len(chunks))
        
        try:
            chunk_text = transcribe_chunk(chunk, i)
            transcriptions.append({
                "chunk_index": i,
                "text": chunk_text
            })
            st.success(f"Successfully transcribed chunk {i+1}")
        except Exception as e:
            st.error(f"Error transcribing chunk {i+1}: {str(e)}")
            transcriptions.append({
                "chunk_index": i,
                "text": f"[Error transcribing chunk: {str(e)}]"
            })
    
    # Combine all transcriptions
    combined_text = " ".join(chunk["text"] for chunk in transcriptions)
    
    # Generate summary
    status.text("Generating summary...")
    summary = summarize_transcript(combined_text)
    
    # Read existing data and append the new transcription
    transcriptions_data = read_transcriptions()
    transcriptions_data.append({
        "file": audio_file.name,
        "full_text": combined_text,
        "chunks": transcriptions,
        "summary": summary
    })
    
    # Write updated data to the JSON file
    write_transcriptions(transcriptions_data)
    
    status.text("Processing completed!")
    progress_bar.progress(1.0)
    
    return combined_text, summary

def display_results(file_name, transcript, summary):
    """Display results for the processed file."""
    with st.expander(f"Results for: {file_name}", expanded=True):
        # Summary section
        st.subheader("Summary")
        st.write(summary)
        
        # Transcript section
        st.subheader("Full Transcript")
        st.write(transcript)

def main():
    st.title("Audio Transcription and Summarization App")
    
    # Add deployment info
    st.sidebar.info("""
    ### Deployment Info
    This version is configured for Streamlit Cloud deployment.
    API key is managed through Streamlit secrets.
    """)
    
    st.write("""
    Upload an audio file to get:
    - Complete transcription
    - Comprehensive summary with key points
    """)
    
    # Single file uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['mp3', 'm4a', 'wav', 'ogg']
    )
    
    if uploaded_file:
        # Display audio player
        st.audio(uploaded_file)
        
        if st.button("Process Audio"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                transcript, summary = process_audio_file(uploaded_file)
                display_results(uploaded_file.name, transcript, summary)
            
            st.success("Processing completed!")
            
            # Show all saved transcriptions
            st.subheader("Transcription History")
            with st.expander("Show complete transcription history"):
                transcriptions = read_transcriptions()
                st.json(transcriptions)

if __name__ == "__main__":
    main() 