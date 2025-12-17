import argparse
import os
import time
import whisper
import torch 

def transcribe_file(filepath, model_size, cpu_threads, device_arg=None):
    """Converts audio file to text and saves as .txt with the same name."""
    
    # 1. Specify a device
    if device_arg:
        device = device_arg
    else: # default device
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model loading ({model_size}) on [{device.upper()}]...")

    # 2. Useful only in CPU-mode
    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    
    # 3. Load model
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Transcribing: {filepath}")
    start_time = time.time()
    
    # 4. Transcribing Process
    use_fp16 = True if device == "cuda" else False
    
    result = model.transcribe(filepath, fp16=use_fp16)
    
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
    parser.add_argument("--model", "-m", default="medium", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size (default: medium)")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of CPUs to use (default: 4)")
    parser.add_argument("--device", "-d", default=None, choices=["cpu", "cuda"], help="Force device (cpu or cuda). Leave empty for auto-detection.")
    
    args = parser.parse_args()

    if os.path.exists(args.file):
        transcribe_file(args.file, args.model, args.threads, args.device)
    else:
        print(f"Error: File not found: {args.file}")

if __name__ == "__main__":
    main()
