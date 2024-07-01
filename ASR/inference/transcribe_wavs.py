#!/usr/bin/env python3
"""Run this script to perform inference with a trained model on multiple audio files and save the results to a CSV.

Authors
 * Peter Plantinga 2024
 * Updated by ChatGPT for multiple files processing
"""
import argparse
import os
import csv
from speechbrain.inference.ASR import EncoderDecoderASR
import torch

def link_file(filename, source_dir, target_dir):
    """Create a symbolic link for file between two directories"""
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    if os.path.lexists(target_path):
        os.remove(target_path)
    os.symlink(source_path, target_path)

def update_progress(progress):
    """Displays or updates a console progress bar.

    Args:
        progress (float): A number between 0 and 1 representing the proportion of progress made.
    """
    bar_length = 40
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1}%".format("#" * block + "-" * (bar_length - block), progress * 100)
    print(text, end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_directory")
    parser.add_argument("save_directory")
    args = parser.parse_args()

    # Setup the directory to save and read model checkpoints
    source_dir = os.path.abspath(args.save_directory)
    target_dir = os.path.dirname(source_dir)
    link_file("model.ckpt", source_dir, target_dir)
    link_file("normalizer.ckpt", source_dir, target_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transcriber = EncoderDecoderASR.from_hparams(
        source=".",
        hparams_file="inference.yaml",
        savedir=target_dir,
        run_opts={"device": str(device)}
    )

        # Process each file and store the results
    results = []
    audio_files = [f for f in os.listdir(args.audio_directory) if f.endswith(".wav")]
    total_files = len(audio_files)
    for index, file in enumerate(audio_files):
        file_path = os.path.join(args.audio_directory, file)
        text = transcriber.transcribe_file(file_path)
        file_no_ext = file.replace(".wav", "")  # Remove the .wav extension
        results.append((file_no_ext, text))  # Use the modified file name
        update_progress((index + 1) / total_files)  # Update progress after each file is processed


    # Output results to CSV
    with open('transcription_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['audio', 'transcript'])
        writer.writerows(results)
    print("\nTranscription completed. Results are saved in 'transcription_results.csv'.")
