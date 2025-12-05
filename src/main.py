from datetime import timedelta
from typing import Optional

from fastapi import FastAPI, Request, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ffmpeg
import numpy as np
import srt as srt
import stable_whisper
from deep_translator import GoogleTranslator

DEFAULT_MAX_CHARACTERS = 80

# Language code mapping for Whisper
WHISPER_LANGUAGE_CODES = {
    'hausa': 'ha',
    'yoruba': 'yo',
    'igbo': 'ig',
    'english': 'en',
    'spanish': 'es',
    'arabic': 'ar',
    'hindi': 'hi',
    'bengali': 'bn',
    'portuguese': 'pt',
    'russian': 'ru',
    'urdu': 'ur',
    'french': 'fr',
    'chinese': 'zh',
    'swahili': 'sw',
}


def get_audio_buffer(filename: str, start: int, length: int):
    """
    input: filename of the audio file, start time in seconds, length of the audio in seconds
    output: np array of the audio data which the model's transcribe function can take as input
    """
    out, _ = (
        ffmpeg.input(filename, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, ss=start, t=length)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_time_stamps(segments: list):
    """
    input: a list of segments from the model's transcribe function
    output: a string of the timestamps and the text of each segment
    """
    string = ""
    for seg in segments:
        string += " ".join([str(seg.start), "->", str(seg.end), ": ", seg.text.strip(), "\n"])
    return string


def split_text_by_punctuation(text: str, max_length: int):
    """Split text into chunks respecting punctuation and max length"""
    chunks = []
    while len(text) > max_length:
        split_pos = max(
            text.rfind(p, 0, max_length) for p in [",", ".", "?", "!", " "] if p in text[:max_length]
        )

        if split_pos == -1:
            split_pos = max_length

        chunks.append(text[:split_pos + 1].strip())
        text = text[split_pos + 1:].strip()

    if text:
        chunks.append(text)

    return chunks


def translate_text(text: str, translate_to: str):
    """Translate text using Google Translator with support for Hausa, Yoruba and other languages"""
    try:
        return GoogleTranslator(source='auto', target=translate_to).translate(text=text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails


def make_srt_subtitles(segments: list, translate_to: str, max_chars: int):
    """Generate SRT subtitles with optional translation"""
    subtitles = []
    for i, seg in enumerate(segments, start=1):
        start_time = seg.start
        end_time = seg.end

        text = (
            translate_text(seg.text.strip(), translate_to)
            if translate_to != "no_translation"
            else seg.text.strip()
        )

        text_chunks = split_text_by_punctuation(text, max_chars)
        duration = (end_time - start_time) / len(text_chunks)

        for j, chunk in enumerate(text_chunks):
            chunk_start = start_time + j * duration
            chunk_end = chunk_start + duration

            subtitle = srt.Subtitle(
                index=len(subtitles) + 1,
                start=timedelta(seconds=chunk_start),
                end=timedelta(seconds=chunk_end),
                content=chunk
            )
            subtitles.append(subtitle)

    return srt.compose(subtitles)


def make_vtt_subtitles(segments: list, translate_to: str, max_chars: int):
    """Generate VTT subtitles with optional translation"""
    vtt_content = "WEBVTT\n\n"
    
    subtitle_index = 1
    for seg in segments:
        start_time = seg.start
        end_time = seg.end

        text = (
            translate_text(seg.text.strip(), translate_to)
            if translate_to != "no_translation"
            else seg.text.strip()
        )

        text_chunks = split_text_by_punctuation(text, max_chars)
        duration = (end_time - start_time) / len(text_chunks)

        for j, chunk in enumerate(text_chunks):
            chunk_start = start_time + j * duration
            chunk_end = chunk_start + duration

            start_td = timedelta(seconds=chunk_start)
            end_td = timedelta(seconds=chunk_end)
            
            # Format timedelta to VTT format (HH:MM:SS.mmm)
            start_str = str(start_td).split('.')[0] + '.' + str(start_td.microseconds // 1000).zfill(3)
            end_str = str(end_td).split('.')[0] + '.' + str(end_td.microseconds // 1000).zfill(3)
            
            vtt_content += f"{subtitle_index}\n"
            vtt_content += f"{start_str} --> {end_str}\n"
            vtt_content += f"{chunk}\n\n"
            subtitle_index += 1

    return vtt_content


def make_txt_transcript(segments: list, translate_to: str, include_timestamps: bool = False):
    """Generate plain text transcript with optional translation and timestamps"""
    lines = []
    
    for seg in segments:
        text = (
            translate_text(seg.text.strip(), translate_to)
            if translate_to != "no_translation"
            else seg.text.strip()
        )
        
        if include_timestamps:
            start_time = str(timedelta(seconds=seg.start)).split('.')[0]
            end_time = str(timedelta(seconds=seg.end)).split('.')[0]
            lines.append(f"[{start_time} -> {end_time}] {text}")
        else:
            lines.append(text)
    
    return "\n\n".join(lines)


def get_whisper_language_code(source_language: str) -> Optional[str]:
    """
    Convert language name to Whisper language code.
    Returns None for auto-detection.
    """
    if source_language == "auto_detect" or not source_language:
        return None
    return WHISPER_LANGUAGE_CODES.get(source_language.lower(), source_language)


app = FastAPI(debug=True)

app.mount('/static', StaticFiles(directory='static'), name='static')
template = Jinja2Templates(directory='templates')


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return template.TemplateResponse('index.html', {"request": request, "text": None})


@app.post('/download/')
async def download_subtitle(
        request: Request,
        file: bytes = File(),
        model_type: str = Form("turbo"),
        timestamps: Optional[str] = Form("False"),
        filename: str = Form("subtitles"),
        file_type: str = Form("srt"),
        max_char: int = Form(DEFAULT_MAX_CHARACTERS),
        source_language: str = Form('auto_detect'),
        task_type: str = Form('transcribe'),
):
    """
    Process audio file and generate subtitles/transcript with optional translation.
    Supports Hausa and Yoruba language transcription via OpenAI Whisper (large-v3 recommended).
    
    Args:
        source_language: Language of the audio (auto_detect, english, hausa, yoruba)
        task_type: transcribe, translate_english, translate_hausa, or translate_yoruba
    """
    
    # Save uploaded audio file
    with open('audio.mp3', 'wb') as f:
        f.write(file)

    # Load Whisper model
    # Use large-v3 or large for best accuracy with Hausa and Yoruba
    print(f"Loading Whisper model: {model_type}")
    model = stable_whisper.load_model(model_type)
    
    # Get language code for Whisper
    whisper_lang = get_whisper_language_code(source_language)
    
    # Configure transcription parameters for better accuracy
    transcribe_params = {
        "regroup": False,
        "beam_size": 5,  # Improved accuracy with beam search
        "best_of": 5,    # Consider multiple candidates
        "patience": 1.0,  # Allow more thorough search
        "no_speech_threshold": 0.1,  # Avoid hallucinated text
        "compression_ratio_threshold": 2.0,  # Filter out poor quality
        "condition_on_previous_text": True,  # Better context understanding
        "task": "transcribe",  # Default to transcribe
    }
    
    # Add language parameter if specified (not auto-detect)
    if whisper_lang:
        transcribe_params["language"] = whisper_lang
        print(f"Transcribing with language: {whisper_lang}")
    else:
        print("Auto-detecting language...")
    
    # Perform transcription with optimized settings
    result = model.transcribe("audio.mp3", **transcribe_params)
    
    # Log detected language
    if hasattr(result, 'language'):
        print(f"Detected language: {result.language}")

    # Determine translation target based on task_type
    if task_type == "transcribe":
        translate_to = "no_translation"
    elif task_type == "translate_english":
        translate_to = "en"
    elif task_type == "translate_hausa":
        translate_to = "ha"
    elif task_type == "translate_yoruba":
        translate_to = "yo"
    else:
        translate_to = "no_translation"

    # Determine output filename
    subtitle_file = f"{filename}.{file_type}"
    
    # Process based on file type
    if file_type == "srt":
        with open(subtitle_file, "w", encoding="utf-8") as f:
            srt_content = make_srt_subtitles(result.segments, translate_to, max_char)
            f.write(srt_content)
            
    elif file_type == "vtt":
        with open(subtitle_file, "w", encoding="utf-8") as f:
            vtt_content = make_vtt_subtitles(result.segments, translate_to, max_char)
            f.write(vtt_content)
            
    elif file_type == "txt":
        with open(subtitle_file, "w", encoding="utf-8") as f:
            # Always include timestamps in TXT format
            txt_content = make_txt_transcript(result.segments, translate_to, include_timestamps=True)
            f.write(txt_content)

    # Return file as download
    media_type = "application/octet-stream"
    response = StreamingResponse(
        open(subtitle_file, 'rb'),
        media_type=media_type,
        headers={'Content-Disposition': f'attachment;filename={subtitle_file}'}
    )

    return response