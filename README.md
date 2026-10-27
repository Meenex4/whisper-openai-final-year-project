## Website which convert speech to text by Whisper model ([Official Repo](https://github.com/openai/whisper))

# Whisper-Based Audio Transcription Web Application

## Overview

This project is a web application for converting audio recordings into text using OpenAI Whisper-based speech recognition. It provides a browser-based interface for submitting audio and generating transcriptions. The backend is implemented with FastAPI, and the application uses Stable-TS (Stable Whisper) to perform transcription and improve timestamp alignment. Transcription results can be exported in supported subtitle and text formats, including SRT, VTT, and TXT. The application also includes optional translation functionality through Google Translator via the `deep-translator` package.

**Source code:** https://github.com/Meenex4/whisper-openai-final-year-project

## Features

- Audio transcription using Whisper-based speech recognition.
- Browser-based interface served locally.
- Subtitle and text output, including SRT, VTT, and TXT.
- Optional text translation using Google Translator.
- Configurable Whisper model selection and transcription parameters.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| Transcription and timestamp alignment | Stable-TS (`stable-ts`) |
| Speech recognition model | OpenAI Whisper model weights, loaded through Stable-TS |
| Audio processing | FFmpeg and `ffmpeg-python` |
| Numerical processing | NumPy |
| HTML templates | Jinja2 |
| Subtitle processing | `srt` |
| Translation | `deep-translator` |

## Requirements

Before installation, ensure that you have:

- Python installed. Python 3.10.0 was used in the development environment; other versions may require compatibility testing.
- Git installed to clone the repository.
- FFmpeg installed and accessible from the system command line.
- An internet connection for installing dependencies and downloading model weights on the first run.
- Sufficient memory and processing capacity to load and run the selected Whisper model.

A compatible PyTorch installation is also required. Performance depends on the selected model and the available CPU or GPU resources.

## Installation

### 1. Clone the repository

Open a terminal or Git Bash and run:

```bash
git clone https://github.com/Meenex4/whisper-openai-final-year-project.git
cd whisper-openai-final-year-project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows Git Bash:

```bash
source venv/Scripts/activate
```

On Windows Command Prompt:

```bat
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install the Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository contains a `requirements.txt` file listing the application's dependencies. For the exact package versions recorded in the development environment, consult `package-versions.txt` in the repository.

### 4. Verify FFmpeg

Check that FFmpeg is installed and available:

```bash
ffmpeg -version
```

If the command is not recognized, install FFmpeg for your operating system and add its executable directory to your system PATH.

## Running the Application

From the repository root, change to the source directory:

```bash
cd src
```

Start the application:

```bash
python run.py
```

The launcher starts the local web server and attempts to open the application in a browser. If the browser does not open automatically, visit:

http://127.0.0.1:8000/

Keep the terminal running while using the application. Stop the server with `Ctrl+C` when finished.

## Model Configuration and Transcription

The application loads a Whisper model through Stable-TS. In the current implementation, the default model selection is `turbo`.

The transcription configuration in `src/main.py` includes the following settings:

| Parameter | Configured value |
|---|---|
| Model selection | `turbo` |
| `regroup` | `False` |
| `beam_size` | `5` |
| `best_of` | `5` |
| `patience` | `1.0` |
| `no_speech_threshold` | `0.1` |
| `compression_ratio_threshold` | `2.0` |
| `condition_on_previous_text` | `True` |
| Task | `transcribe` |

These values describe the current source-code configuration. They should not be interpreted as evidence of optimal performance across all audio datasets or recording conditions.

The application also uses FFmpeg for audio preprocessing. The current implementation converts audio to mono, 16 kHz, signed 16-bit PCM for the relevant processing stage.

## Output Formats

The application supports transcription output in the following formats:

- **TXT:** Plain-text transcription.
- **SRT:** SubRip subtitles with timestamps.
- **VTT:** Web Video Text Tracks subtitles.

The generated output depends on the submitted audio, selected model, and transcription configuration.

## Repository Structure

The principal files and directories include:

```text
whisper-openai-final-year-project/
├── src/
│   ├── main.py
│   └── run.py
├── requirements.txt
├── package-versions.txt
├── Dockerfile
├── LICENSE
└── README.md
```

- `src/main.py` contains the FastAPI application and transcription-related functionality.
- `src/run.py` provides the local application launcher.
- `requirements.txt` lists Python dependencies.
- `package-versions.txt` records the package versions captured from the development environment.
- `Dockerfile` contains a container build configuration.
- `LICENSE` contains the repository's license information.

Additional source files and interface resources may be present in the repository.

## Reproducibility and Evaluation

The source code and dependency information are publicly available in this repository. The model selection and transcription parameters are documented above to support inspection and independent setup.

For reproducible evaluation, researchers should record the Python version, operating system, FFmpeg version, installed package versions, model selected, transcription parameters, and audio preprocessing settings. Evaluation datasets should have documented provenance and appropriate permission for use and sharing.

** Evaluation scripts and consent-safe sample audio:** These should be added to the repository before claiming that a complete, independently executable evaluation package is available. Until those materials are included and tested, independent evaluation may require researchers to prepare suitable test audio and evaluation procedures.


## License

See the [`LICENSE`](LICENSE) file for the license applicable to this repository.

## Hosting website on localhost:

1. Clone the repo - `git clone https://github.com/Kabanosk/whisper-website.git`
2. Go to repo directory - `cd whisper-website`
3. Create virtual environment - `python3 -m venv venv`
4. Activate the environment - `source venv/bin/activate`/`. venv/bin/activate`
5. Install requirements - `pip install -r requirements.txt`
6. Go to src directory - `cd src`
7. Run the `run.py` file - `python3 run.py`
8. Go to your browser and type `http://127.0.0.1:8000/` if the browser doesn't open

## Run website on localhost with Docker
### First time
1. Install [Docker](https://docs.docker.com/engine/install/)
2. Clone the repo - `git clone https://github.com/Kabanosk/whisper-website.git`
3. Go to repo directory - `cd whisper-website`
4. Create Docker image - `docker build -t app .`
5. Run Docker container - `docker run --name app_container -p 80:80 app`
6. Go to your browser and type `http://127.0.0.1:80/`

### Next time

1. Start your Docker container - `docker start app_container`
2. Go to your browser and type `http://127.0.0.1:80/`
