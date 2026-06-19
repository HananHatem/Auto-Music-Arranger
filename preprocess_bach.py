"""
Bach Chorales Processor - Unified Version

Combines functionality from:
- prepare.py (JSON to numpy arrays)
- preprocess.py (MXL/XML to JSON with music21)
- collect_bach.py (MXL/XML processing with muspy)
"""

import argparse
import logging
import operator
import random
import json
from pathlib import Path
from collections import defaultdict
from operator import itemgetter

import joblib
import numpy as np
import tqdm
import music21.converter
import muspy

# ================================================
# Configuration
# ================================================
TRACK_NAMES = ["Soprano", "Alto", "Tenor", "Bass"]
PROGRAMS = {
    "Soprano": 52,
    "Alto": 53,
    "Tenor": 53,
    "Bass": 32
}

SPECIAL_CASES = {
    "bwv171.6": {
        "Soprano\rOboe 1,2\rViolin1": "Soprano",
        "Alto\rViloin 2": "Alto",
        "Tenor\rViola": "Tenor",
        "Bass\rContinuo": "Bass"
    },
    "bwv41.6": {
        "Soprano Oboe 1 Violin1": "Soprano",
        "Alto Oboe 2 Viloin 2": "Alto",
        "Tenor Viola": "Tenor",
        "Bass Continuo": "Bass"
    }
}

# ================================================
# Helper Functions
# ================================================
def setup_logging(output_dir=None, quiet=False):
    """Configure logging system."""
    level = logging.WARNING if quiet else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if output_dir:
        log_file = output_dir / "processing.log"
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Bach chorales.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Parser for music files processing
    music_parser = subparsers.add_parser('process_music', help='Process MXL/XML music files')
    music_parser.add_argument("-i", "--input_dir", type=Path, required=True,
                            help="Directory containing music files (.mxl, .xml)")
    music_parser.add_argument("-o", "--output_dir", type=Path, required=True,
                            help="Output directory for processed files")
    music_parser.add_argument("-j", "--n_jobs", type=int, default=1,
                            help="Number of parallel jobs")
    music_parser.add_argument("-q", "--quiet", action="store_true",
                            help="Reduce output verbosity")
    
    # Parser for JSON preparation
    json_parser = subparsers.add_parser('prepare_json', help='Prepare JSON files for training')
    json_parser.add_argument("-i", "--input_dir", type=Path, required=True,
                            help="Input data directory (JSON files)")
    json_parser.add_argument("-o", "--output_dir", type=Path, required=True,
                            help="Output directory")
    json_parser.add_argument("-j", "--n_jobs", type=int, default=1,
                            help="Number of workers")
    json_parser.add_argument("-q", "--quiet", action="store_true",
                            help="Reduce output verbosity")
    
    return parser.parse_args()

def identify_track(track_name, filename):
    """Identify the voice part from track name."""
    filename = filename.stem
    
    # Check special cases first
    if filename in SPECIAL_CASES:
        return SPECIAL_CASES[filename].get(track_name)
    
    # Normal case - match track name
    lower_name = track_name.lower()
    for name in TRACK_NAMES:
        if name.lower() in lower_name:
            return name
    
    return None

def process_notes(notes):
    """Process and clean note list."""
    unique_notes = {(n.offset, n.pitch, n.duration.quarterLength): n for n in notes}
    return sorted(unique_notes.values(),
                 key=lambda x: (x.offset, x.pitch.ps, x.duration.quarterLength))

def convert_to_serializable(note):
    """Convert music21 note to serializable format."""
    return {
        "offset": float(note.offset),
        "pitch": int(note.pitch.ps),
        "duration": float(note.duration.quarterLength),
        "volume": getattr(note.volume, 'velocity', 64)
    }

# ================================================
# Music Files Processing (MXL/XML to JSON)
# ================================================
def process_music_file(filename):
    """Process a single music file to JSON format."""
    try:
        score = music21.converter.parse(filename)
        voice_parts = defaultdict(list)
        
        for part in score.parts:
            track_name = part.partName or ""
            voice_part = identify_track(track_name, filename)
            
            if voice_part is None:
                continue
                
            for note in part.flat.notes:
                if note.isNote:
                    voice_parts[voice_part].append(note)
                elif note.isChord:
                    highest_note = max(note, key=lambda x: x.pitch.ps)
                    voice_parts[voice_part].append(highest_note)
        
        processed_parts = {}
        for part_name, notes in voice_parts.items():
            processed_parts[part_name] = process_notes(notes)
        
        if sum(len(notes) > 10 for notes in processed_parts.values()) < 2:
            return None
        
        output = {
            "metadata": {
                "title": filename.stem,
                "source": "Bach Chorales",
                "parts": list(processed_parts.keys())
            },
            "tracks": []
        }
        
        for part_name, notes in processed_parts.items():
            track = {
                "name": part_name,
                "program": PROGRAMS[part_name],
                "is_drum": False,
                "notes": [convert_to_serializable(n) for n in notes]
            }
            output["tracks"].append(track)
            
        return output
        
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return None

def process_music_files(args):
    """Process all music files in the input directory."""
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for split in ("train", "valid", "test"):
        (args.output_dir / split).mkdir(exist_ok=True)
    
    input_files = list(args.input_dir.glob("*.mxl")) + list(args.input_dir.glob("*.xml"))
    if not input_files:
        logging.error("No input files found in the specified directory.")
        return
    
    random.seed(42)
    random.shuffle(input_files)
    total_files = len(input_files)
    train_end = int(0.8 * total_files)
    valid_end = train_end + int(0.1 * total_files)
    
    splits = []
    for i in range(total_files):
        if i < train_end:
            splits.append("train")
        elif i < valid_end:
            splits.append("valid")
        else:
            splits.append("test")
    
    def process_and_save(filename, split):
        data = process_music_file(filename)
        if data is None:
            return None
            
        output_path = args.output_dir / split / f"{filename.stem}.json"
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            return filename
        except Exception as e:
            logging.error(f"Error saving {output_path}: {str(e)}")
            return None
    
    success_count = 0
    if args.n_jobs == 1:
        for filename, split in tqdm.tqdm(zip(input_files, splits), 
                                      total=len(input_files),
                                      disable=args.quiet):
            if process_and_save(filename, split):
                success_count += 1
    else:
        results = joblib.Parallel(args.n_jobs, verbose=0 if args.quiet else 5)(
            joblib.delayed(process_and_save)(filename, split)
            for filename, split in zip(input_files, splits)
        )
        success_count = sum(1 for r in results if r is not None)
    
    logging.info(f"Successfully processed {success_count}/{total_files} files.")

# ================================================
# JSON Preparation (JSON to numpy arrays)
# ================================================
def get_arrays(notes, labels, n_tracks, seq_len):
    """Convert notes and labels to numpy arrays."""
    data = {
        "time": np.zeros((seq_len,), int),
        "pitch": np.zeros((seq_len,), int),
        "duration": np.zeros((seq_len,), int),
        "velocity": np.zeros((seq_len,), int),
        "label": np.zeros((seq_len,), int),
        "onset_hint": np.zeros((n_tracks,), int),
        "pitch_hint": np.zeros((n_tracks,), int),
    }

    for i, (note, label) in enumerate(zip(notes, labels)):
        data["time"][i] = int(note[0])
        data["pitch"][i] = int(note[1]) + 1  # +1 to reserve 0
        data["duration"][i] = int(note[2] * 4)  # e.g. quarterLength to ticks
        data["velocity"][i] = int(note[3])
        data["label"][i] = label + 1  # +1 to reserve 0

    for i in range(n_tracks):
        nonzero = (data["label"] == i + 1).nonzero()[0]
        if nonzero.size:
            data["onset_hint"][i] = nonzero[0]
            data["pitch_hint"][i] = round(np.mean(data["pitch"][nonzero]))

    return data

def process_json_file(filename):
    """Process a single JSON file into arrays."""
    try:
        with open(filename) as f:
            score = json.load(f)

        notes, labels = [], []
        
        if "tracks" not in score:
            logging.error(f"Missing 'tracks' in {filename}")
            return None

        for track in score["tracks"]:
            if track["is_drum"]:
                continue
            name = track["name"]
            if name not in TRACK_NAMES:
                continue
            label = TRACK_NAMES.index(name)
            
            if "notes" not in track or not track["notes"]:
                logging.error(f"Missing or empty 'notes' in track '{name}' of {filename}")
                continue

            for note in track["notes"]:
                if None in (note.get("offset"), note.get("pitch"), note.get("duration"), note.get("volume")):
                    continue
                notes.append((note["offset"], note["pitch"], note["duration"], note["volume"]))
                labels.append(label)

        if not notes:
            logging.error(f"No valid notes in {filename}")
            return None

        notes, labels = zip(*sorted(zip(notes, labels), key=itemgetter(0)))
        arrays = get_arrays(notes, labels, len(TRACK_NAMES), len(notes))
        return arrays

    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return None

def prepare_json_files(args):
    """Prepare JSON files for training."""
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    features = ["time", "pitch", "duration", "velocity", "label", "onset_hint", "pitch_hint"]
    subsets = ("train", "valid", "test")

    for subset in subsets:
        filenames = list((args.input_dir / subset).glob("*.json"))
        if not filenames:
            logging.warning(f"No files found in {subset}/")
            continue

        if args.n_jobs == 1:
            data = []
            for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80):
                result = process_json_file(filename)
                if result:
                    data.append({"filename": filename.stem, "arrays": result})
        else:
            results = joblib.Parallel(args.n_jobs, verbose=0 if args.quiet else 5)(
                joblib.delayed(process_json_file)(filename) for filename in filenames
            )
            data = [
                {"filename": filename.stem, "arrays": result}
                for filename, result in zip(filenames, results) if result
            ]

        if subset in ("valid", "test"):
            data.sort(key=lambda x: len(x["arrays"]["time"]))

        for name in features:
            np.savez(
                args.output_dir / f"{name}_{subset}.npz",
                *[sample["arrays"][name] for sample in data],
            )

        with open(args.output_dir / f"filenames_{subset}.txt", "w") as f:
            for sample in data:
                f.write(f"{sample['filename']}\n")

        logging.info(f"Saved {len(data)} samples for {subset}")

# ================================================
# Main Function
# ================================================
def main():
    args = parse_arguments()
    setup_logging(args.output_dir if hasattr(args, 'output_dir') else None, args.quiet)
    
    if args.command == "process_music":
        process_music_files(args)
    elif args.command == "prepare_json":
        prepare_json_files(args)

if __name__ == "__main__":
    main()