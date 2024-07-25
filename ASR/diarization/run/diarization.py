from pyannote.audio import Pipeline
import torch
import torchaudio
import os
import pandas as pd
from speechbrain.inference.ASR import EncoderDecoderASR

# Constants
HF_TOKEN = "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"
MODEL_DIR = "../model"
AUDIO_DIR = "../data/wavs"
OUTPUT_DIR = "../segments"
RESULTS_OUTPUT_DIR = "../output"
LOCAL_MODEL_PATH = "../../results/CRDNN_BPE_960h_LM/2602/save/CKPT+2024-07-01+14-24-57+00"

# Load diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN, cache_dir=MODEL_DIR)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# Load ASR model
asr_model = EncoderDecoderASR.from_hparams(
    source=LOCAL_MODEL_PATH,
    hparams_file="inference.yaml",
    savedir="tmpdir",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Process each audio file in the directory
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, filename)
        
        # Apply pretrained pipeline on audio
        diarization = pipeline(audio_path)
        
        # Load the original audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}segment{turn.start:.1f}_{turn.end:.1f}.wav")
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            torchaudio.save(segment_path, waveform[:, start_sample:end_sample], sample_rate)
            if (turn.end - turn.start) > 0.17:
                transcription = asr_model.transcribe_file(segment_path)
            else:
                transcription = ""
            speaker_number = int(speaker.split('_')[-1])
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker_number,
                "text": transcription
            })

            os.remove(segment_path)
        
        # Save results to a JSON file
        df = pd.DataFrame(results)
        result_file = os.path.splitext(filename)[0] + '.json'
        full_path = os.path.join(RESULTS_OUTPUT_DIR, result_file)
        df.to_json(full_path, orient="records", lines=False, force_ascii=False, indent=4)

        print(f"Results saved to {full_path}")