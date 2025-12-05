import os
import time
import argparse
import whisper
import torch

def transcribe_file(filepath, model_size, cpu_threads):
    """Converts audio file to text and saves as .txt with the same name."""
    
    print(f"Model loading ({model_size})...")
    
    # Limiting the number of threads for CPU processing
    torch.set_num_threads(cpu_threads)
    
    # Loading the model on CPU (use "cuda" if GPU is available)
    model = whisper.load_model(model_size, device="cpu")
    
    print(f"Transcribing: {filepath}")
    start_time = time.time()
    
    # Transcribe the audio file
    # fp16=False is necessary to avoid warnings when running on CPU.
    result = model.transcribe(filepath, fp16=False)
    
    duration = time.time() - start_time
    print(f"Done! Elapsed Time: {duration:.2f} seconds")
    
    # Save the transcription to a .txt file
    output_file = os.path.splitext(filepath)[0] + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"Saved: {output_file}")

def main():
    
    # OpenAI Whisper transcription tool argument parsing
    parser = argparse.ArgumentParser(description="OpenAI Whisper Transcription Tool")
    
    parser.add_argument("file", help="Video/Audio file path to transcribe")
    parser.add_argument("--model", "-m", default="small", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size to use (default: small)")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of CPUs to use (default: 4)")
    args = parser.parse_args()

    if os.path.exists(args.file):
        transcribe_file(args.file, args.model, args.threads)
    else:
        print(f"Error: File not found: {args.file}")


if __name__ == "__main__":
    
    main()