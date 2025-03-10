import os
import json
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("API Key not found. Please set it in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Define the JSON file to store transcriptions
JSON_FILE = "transcriptions.json"

# System prompt for GPT-4 summarization
SUMMARY_SYSTEM_PROMPT = """You are an expert meeting summarizer. Your task is to analyze meeting transcripts and create comprehensive summaries that capture the essential information. Please provide your summary in the following format:

1. Key Points:
   - List the main topics and decisions discussed
   - Highlight any important announcements or changes
   - Note any action items or deadlines mentioned

2. Discussion Topics:
   - Break down major discussion points
   - Include relevant context and explanations
   - Capture different perspectives if any were presented

3. Conclusions & Next Steps:
   - Summarize final decisions or consensus reached
   - List agreed-upon action items with owners (if mentioned)
   - Note any scheduled follow-ups or future meetings

4. Additional Notes:
   - Include any important details that don't fit above categories
   - Highlight any unresolved issues or points requiring further discussion
   - Mention any notable quotes or important statements

Please ensure your summary is:
- Clear and concise while retaining important details
- Organized logically with proper categorization
- Written in professional language
- Focused on actionable insights and key takeaways"""

# Function to read existing transcriptions
def read_transcriptions():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  # Return an empty list if file is empty or invalid
    return []

# Function to write new transcription data
def write_transcriptions(data):
    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def split_audio(audio_path, chunk_duration_ms=600000):  # 10 minutes in milliseconds
    """Split audio file into chunks of specified duration."""
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    
    duration = len(audio)
    chunks = []
    
    # Calculate number of chunks
    num_chunks = (duration + chunk_duration_ms - 1) // chunk_duration_ms
    print(f"Audio duration: {duration/1000:.2f} seconds")
    print(f"Splitting into {num_chunks} chunks...")
    
    for i in range(num_chunks):
        start_time = i * chunk_duration_ms
        end_time = min((i + 1) * chunk_duration_ms, duration)
        chunk = audio[start_time:end_time]
        chunks.append(chunk)
        print(f"Created chunk {i+1}/{num_chunks} ({len(chunk)/1000:.2f} seconds)")
    
    return chunks

def transcribe_chunk(chunk, chunk_index):
    """Transcribe a single audio chunk."""
    # Create a temporary file for the chunk
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = temp_file.name
        try:
            # Export chunk to temporary file
            chunk.export(temp_path, format="mp3")
            
            # Transcribe the chunk
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

def summarize_transcript(transcript):
    """Generate a summary of the transcript using GPT-4."""
    print("\nGenerating meeting summary using GPT-4...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        print(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def process_audio_file(audio_path):
    """Process an audio file by splitting it into chunks and transcribing each chunk."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    
    # Split audio into chunks
    chunks = split_audio(audio_path)
    
    # Transcribe each chunk
    transcriptions = []
    for i, chunk in enumerate(chunks):
        print(f"\nTranscribing chunk {i+1}/{len(chunks)}...")
        try:
            chunk_text = transcribe_chunk(chunk, i)
            transcriptions.append({
                "chunk_index": i,
                "text": chunk_text
            })
            print(f"Successfully transcribed chunk {i+1}")
        except Exception as e:
            print(f"Error transcribing chunk {i+1}: {str(e)}")
            transcriptions.append({
                "chunk_index": i,
                "text": f"[Error transcribing chunk: {str(e)}]"
            })
    
    # Combine all transcriptions
    combined_text = " ".join(chunk["text"] for chunk in transcriptions)
    
    # Generate summary using GPT-4
    summary = summarize_transcript(combined_text)
    
    # Read existing data and append the new transcription
    transcriptions_data = read_transcriptions()
    transcriptions_data.append({
        "file": os.path.basename(audio_path),
        "full_text": combined_text,
        "chunks": transcriptions,
        "summary": summary
    })
    
    # Write updated data to the JSON file
    write_transcriptions(transcriptions_data)
    print(f"\nTranscription and summary completed and saved for: {audio_path}")
    print("\nSummary of the meeting:")
    print("=" * 50)
    print(summary)
    print("=" * 50)

if __name__ == "__main__":
    # Get audio file path
    audio_path = "Exercise is medicine.m4a"
    process_audio_file(audio_path)
