# 🎙️ Whisper Transcriber CLI

Whisper Transcriber CLI is a specialized Python tool designed to run OpenAI's state-of-the-art speech recognition models on legacy and cooling-constrained hardware.

Standard Whisper implementations often utilize 100% of available CPU resources, which can lead to thermal throttling or overheating on older systems (such as 1st Gen Intel Core i7s). This tool solves that problem by providing granular control over CPU thread usage, allowing users to balance transcription speed with hardware safety. It offers a simple command-line interface for processing local audio/video files with custom output naming and model selection.

> **A hardware-friendly command-line tool for transcribing audio using OpenAI's Whisper model.**

This tool is a Python wrapper around [OpenAI Whisper](https://github.com/openai/whisper) designed with **legacy hardware** and **temperature management** in mind. It allows users to limit CPU thread usage to prevent overheating on older processors (e.g., 1st Gen Intel i7) while delivering state-of-the-art transcription accuracy.

## ⚡ Features

  - **🔥 CPU Protection:** Manually limit the number of CPU threads to prevent 100% load and overheating on older systems.
  - **🧠 Model Flexibility:** Support for all Whisper models (`tiny`, `base`, `small`, `medium`, `large`) to balance speed vs. accuracy.
  - **📝 Custom Output:** Easily specify the output filename for your transcripts.
  - **🛠️ Simple CLI:** Clean and easy-to-use command-line interface using `argparse`.
  - **🖥️ CPU Optimized:** Configured to run on CPU by default (no CUDA/GPU requirement), making it accessible for any machine.

## 📦 Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python 3.8** or higher.
2.  **FFmpeg**: Required by Whisper to process audio files.
      * **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
      * **Windows:** Install via [Chocolatey](https://chocolatey.org/) (`choco install ffmpeg`) or download from [ffmpeg.org](https://ffmpeg.org/).
      * **MacOS:** `brew install ffmpeg`

## 🚀 Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/whisper-transcriber.git
    cd whisper-transcriber
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install openai-whisper torch
    ```

## 📖 Usage

Run the script from your terminal providing the audio file as an argument.

```bash
python main.py [AUDIO_FILE] [OPTIONS]
```

### Arguments

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | - | **Required** | Path to the input audio/video file. |
| `--model` | `-m` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). |
| `--threads`| `-t` | `4` | Number of CPU threads to use. Lower this value to reduce CPU load/heat. |
| `--output` | `-o` | `None` | Custom name for the output `.txt` file (without extension). |

### 💡 Examples

**1. Basic Transcription (Default Settings):**
Uses the `small` model and limits to 4 threads.

```bash
python main.py interview.wav
```

**2. High Accuracy Mode:**
Uses the `medium` model for better precision (slower).

```bash
python main.py lecture.mp3 --model medium
```

**3. Safe Mode for Old Hardware (Low Heat):**
Uses only 2 threads to keep the CPU cool.

```bash
python main.py long_recording.mkv --threads 2
```

**4. Custom Output Filename:**
Saves the transcript as `meeting_notes.txt` instead of `recording_001.txt`.

```bash
python main.py recording_001.wav --output meeting_notes
```

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

-----

**Disclaimer:** This project uses OpenAI's Whisper model. Please refer to OpenAI's [Model Card](https://github.com/openai/whisper/blob/main/model-card.md) for more details on limitations and usage.