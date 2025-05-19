import json
import os
import shutil
import time
from datetime import datetime

import pandas as pd
import yaml

from .globals import LOG_FILENAME, METADATA_FILENAME, TIMESTAMP_FORMAT, TRANSCRIPT_DIR


def create_directory(file_id):
    """Creates a directory for storing transcription files."""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    file_directory = os.path.join(TRANSCRIPT_DIR, file_id)
    os.makedirs(file_directory, exist_ok=True)


def create_file_id(file_path, timestamp):
    """Creates a unique identifier for a file composed of the file path and timestamp."""
    # Extract filename from file_path
    file_base_name = os.path.basename(file_path)
    # Use rsplit to split from the right at most once

    timestamp = timestamp.replace(" ", "-").replace("-", "")
    timestamp = timestamp[:-2]
    timestamp = timestamp[2:]

    short_base_name = (
        file_base_name[0:7] if len(file_base_name) >= 5 else file_base_name
    )
    file_id = timestamp + "-" + short_base_name
    return file_id


def create_output_files(result, speaker_detection, file_id):
    """Creates only the plain text transcription file."""
    create_txt_file(result, file_id, speaker_detection, timestamps=False, maxqda=False)


def create_txt_file(result, file_id, speaker_detection, timestamps, maxqda):
    """Creates a TXT file for the transcription result, grouping text by speaker."""
    segments = result["segments"]
    filename = "transcription.txt"
    
    file_path = os.path.join(TRANSCRIPT_DIR, file_id, filename)
    
    with open(file_path, "w", encoding="utf-8") as file:
        headline = (
            f"Transcription for {file_id}"
            + ("" if maxqda and speaker_detection else "\n")
            + ("" if speaker_detection else "\n")
        )
        file.write(headline)
        
        # Group segments by speaker
        if speaker_detection:
            # Initialize variables
            current_speaker = None
            speaker_texts = []
            
            # Process all segments
            for segment in segments:
                speaker = segment["speaker"] if "speaker" in segment else "Speaker undefined"
                text = str(segment["text"]).lstrip()
                
                if timestamps:
                    start_time = time.strftime("[%H:%M:%S]", time.gmtime(segment["start"]))
                    text = f"{start_time} - {text}"
                
                # If this is a new speaker, write the previous speaker's text
                if speaker != current_speaker:
                    # Write the previous speaker's text if it exists
                    if current_speaker is not None and speaker_texts:
                        file.write(("\n\n" if maxqda else "\n") + current_speaker + ": ")
                        # Join all the speaker's text with spaces instead of newlines
                        combined_text = " ".join(speaker_texts)
                        file.write(combined_text + (" " if maxqda else "\n"))
                    
                    # Reset for the new speaker
                    current_speaker = speaker
                    speaker_texts = [text]
                else:
                    # Add this text to the current speaker
                    speaker_texts.append(text)
            
            # Write the last speaker's text
            if current_speaker is not None and speaker_texts:
                file.write(("\n\n" if maxqda else "\n") + current_speaker + ": ")
                combined_text = " ".join(speaker_texts)
                file.write(combined_text + (" " if maxqda else "\n"))
        else:
            # If no speaker detection, just write each segment on its own line
            for segment in segments:
                text = str(segment["text"]).lstrip()
                if timestamps:
                    start_time = time.strftime("[%H:%M:%S]", time.gmtime(segment["start"]))
                    text = f"{start_time} - {text}"
                file.write(text + (" " if maxqda else "\n"))




def transform_speakers_results(diarization_segments):
    """Transforms diarization segments to speaker results."""

    diarize_df = pd.DataFrame(diarization_segments.itertracks(yield_label=True))
    diarize_df["start"] = diarize_df[0].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df[0].apply(lambda x: x.end)
    diarize_df.rename(columns={2: "speaker"}, inplace=True)
    return diarize_df


def named_tuple_to_dict(obj):
    """Converts named tuple to dictionary."""
    if isinstance(obj, dict):
        return {key: named_tuple_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [named_tuple_to_dict(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: named_tuple_to_dict(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(named_tuple_to_dict(value) for value in obj)
    else:
        return obj


def isnamedtupleinstance(x):
    """Checks if the object is an instance of namedtuple."""
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i) == str for i in fields)


def create_metadata(
    file_id,
    filename,
    audio_duration,
    model,
    language,
    speaker_detection,
    num_speakers,
    device,
    compute_type,
    timestamp,
    original_audio_filename,
):
    """Creates metadata file for the transcription."""

    metadata_file_path = os.path.join(TRANSCRIPT_DIR, file_id, METADATA_FILENAME)
    metadata = {
        "file_id": file_id,
        "filename": filename,
        "audio_duration": audio_duration,
        "model": model,
        "language": language,
        "speaker_detection": speaker_detection,
        "num_speakers": num_speakers,
        "device": device,
        "compute_type": compute_type,
        "timestamp": timestamp,
        "path_to_audio_file": original_audio_filename,
    }
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
        yaml.dump(metadata, metadata_file)


def write_logfile(message, file_id):
    """Writes a log message to the log file."""

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    log_file_path = os.path.join(TRANSCRIPT_DIR, file_id, LOG_FILENAME)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ------ {message}\n")


def add_processing_time_to_metadata(file_id):
    """Adds processing time information to metadata."""

    metadata_file_path = os.path.join(TRANSCRIPT_DIR, file_id, METADATA_FILENAME)
    with open(metadata_file_path, "r", encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    timestamp = metadata["timestamp"]
    start_time = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    stop_time = timestamp = datetime.now()
    processing_time = stop_time - start_time
    metadata["processing_time"] = int(processing_time.total_seconds())
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
        yaml.dump(metadata, metadata_file)


def delete_transcription(file_id):
    """Deletes the transcription files."""

    file_id = "" if file_id == "all" else file_id
    directory_name = os.path.join(TRANSCRIPT_DIR, file_id)
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
