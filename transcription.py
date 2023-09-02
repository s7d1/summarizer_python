import os

import torch
import torchaudio
from pydub import AudioSegment
from torchaudio.transforms import Resample
from transformers import pipeline

from hf_helpers import print_gpu_utilization
from ytdl import download_audio


def whisper_pipeline(speech_model_identifier="openai/whisper-small"):
    """
    Initializes the Whisper speech model pipeline.

    Args:
        speech_model_identifier (str): The identifier or path of the speech model.

    Returns:
        speech_model_pipeline
    """
    transcriber = pipeline(task="automatic-speech-recognition", model=speech_model_identifier, device_map="auto", use_fast=True, chunk_length_s=30)
    return transcriber


def break_audio_into_chunks(audio_path, chunk_duration=30000):
    """
    Breaks the audio file into sequential chunks of up to the specified duration (default: 30 seconds).
    Returns a list of paths for the generated audio chunks.

    Args:
        audio_path (str): Path of the input audio file.
        chunk_duration (int): Duration of each audio chunk in milliseconds. Default is 30000ms (30 seconds).

    Returns:
        list: List of paths for the generated audio chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio)
    num_chunks = (total_duration // chunk_duration) + 1 if total_duration % chunk_duration != 0 else (total_duration // chunk_duration)
    chunks_paths = []

    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)
        chunk_audio = audio[start_time:end_time]

        chunk_path = f"chunk_{i+1}_{audio_path}"
        chunk_audio.export(chunk_path, format="wav")
        chunks_paths.append(chunk_path)

    return chunks_paths


def resample_audio(audio_path, target_sr=16000):
    """
    Resamples an audio file to a target sampling rate and saves the resampled audio as a new WAV file.

    Args:
        audio_path (str): Path to the input audio file.
        target_sr (int): Target sampling rate in Hz. Default is 16000.

    Returns:
        str: Path to the resampled audio file.

    Raises:
        ValueError: If the input audio file is not found or cannot be loaded.
        ValueError: If the target sampling rate is invalid or unsupported.
        RuntimeError: If the resampled audio file cannot be saved.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except FileNotFoundError:
        raise ValueError("Input audio file not found or cannot be loaded.")

    resampler = Resample(orig_freq=sample_rate, new_freq=target_sr)
    resampled_waveform = resampler(waveform)

    dirname = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)
    new_path = os.path.join(dirname, "resampled_" + filename)

    try:
        torchaudio.save(new_path, resampled_waveform, sample_rate=target_sr)
    except RuntimeError:
        raise RuntimeError("Failed to save the resampled audio file.")

    return new_path


def convert_audio_to_mono(audio_path):
    """
    Converts the audio file to mono channel and returns the path of the mono audio file.

    Args:
        audio_path (str): Path of the input audio file.

    Returns:
        str: Path of the mono audio file.
    """
    mono_audio_path = f"mono_{audio_path}"
    audio = AudioSegment.from_file(audio_path)
    mono_audio = audio.set_channels(1)
    mono_audio.export(mono_audio_path, format="wav")
    return mono_audio_path


def transcribe(video_link, results):
    """
    Transcribes the audio from a video and appends the result to the provided results list.

    Args:
        video_link (str): The link of the video to transcribe. The video must be downloadable by youtube-dl (ytdl).
        results (list): The list to store the transcriptions.

    Raises:
        FileNotFoundError: If the audio file or any intermediate files cannot be found.
        Exception: If any error occurs during the transcription process.

    Note:
        This function assumes the availability of the following helper functions from the appropriate modules:
        - download_audio(video_link): Downloads the audio from the video.
        - convert_audio_to_mono(audio_file_path): Converts the audio file to mono channel.
        - resample_audio(mono_audio_file_path, target_sample_rate): Resamples the audio to the specified sample rate.
        - transcribe_audio(audio_file_path, model, processor): Performs audio transcription using a specified model and processor.

    This function transcribes the audio from a video by following these steps:
    1. Downloads the audio from the provided video link using youtube-dl (ytdl).
    2. Converts the audio file to mono channel for better transcription accuracy.
    3. Removes the original audio file.
    4. Initializes the transcription pipeline.
    5. Performs audio transcription using the initialized pipeline.
    6. Retrieves the transcriptions and combines them into a single text.
    7. Removes the intermediate audio files.
    8. Frees GPU memory used by the pipeline and clears the cache.
    9. Appends the transcribed text to the results list.

    Example usage:
        results = []
        video_link = "https://example.com/video"
        transcribe(video_link, results)
        print(results)  # Contains the transcribed text from the video.
    """

    try:
        audio_file_path = download_audio(video_link)
        # chunks = break_audio_into_chunks(audio_file_path)
        mono_audio_file_paths = [convert_audio_to_mono(audio_file_path)]
        os.remove(audio_file_path)
        # mono_audio_file_paths = [convert_audio_to_mono(chunk) for chunk in chunks]
        """for file in chunks:
            os.remove(file)
        """
        pipe = whisper_pipeline()

        # resampled_audio_file_path = resample_audio(mono_audio_file_paths, 16000)
        transcription = pipe(mono_audio_file_paths)
        # pipe(resampled_audio_file_path)
        print_gpu_utilization()
        print(transcription)
        transcription_text = " ".join([item["text"] for item in transcription])
        print(transcription_text)
        for file in mono_audio_file_paths:
            os.remove(file)
        # os.remove(resampled_audio_file_path)

        # Free model and processor memory on the GPU
        # del model
        # del processor
        del pipe
        torch.cuda.empty_cache()
        results.append(transcription_text)

    except FileNotFoundError as e:
        # Handle missing file error
        print(f"Error: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"Error: {e}")

