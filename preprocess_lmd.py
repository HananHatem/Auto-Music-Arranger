"""
Enhanced LMD MIDI Processor with Robust Error Handling - Complete Version
"""

import argparse
import logging
import random
import json
import gzip
import os
from pathlib import Path
from collections import defaultdict
import traceback

import joblib
import tqdm
import muspy
from mido import MidiFile, MidiTrack, Message

# Configuration
LMD_CONFIG = {
    "programs": {
        "Piano": 0,
        "Guitar": 1,
        "Bass": 2,
        "Strings": 3,
        "Brass": 4,
        "Drums": 5
    },
    "resolution": 24,
    "max_duration": 1200,
    "max_beat": 4096,
    "max_duration_ticks": 192
}

failed_files = []

def setup_logging(output_dir=None, quiet=False):
    """Configure comprehensive logging system."""
    level = logging.WARNING if quiet else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        log_file = output_dir / "processing.log"
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments():
    """Parse command line arguments with enhanced validation."""
    parser = argparse.ArgumentParser(description='Process LMD MIDI files with nested directory support.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # MIDI Processing Parser
    midi_parser = subparsers.add_parser('process_midi', help='Process nested MIDI directories')
    midi_parser.add_argument("-i", "--input_dir", type=Path, required=True,
                          help="Root directory containing nested MIDI folders")
    midi_parser.add_argument("-o", "--output_dir", type=Path, required=True,
                          help="Output directory for processed JSON files")
    midi_parser.add_argument("--skip_errors", action="store_true",
                          help="Skip corrupted files without repair attempts")
    midi_parser.add_argument("--max_samples", type=int, default=None,
                          help="Maximum number of files to process")
    midi_parser.add_argument("-j", "--n_jobs", type=int, default=1,
                          help="Number of parallel processing jobs")
    midi_parser.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress informational output")
    midi_parser.add_argument("--min_notes", type=int, default=10,
                          help="Minimum notes required per track")
    
    # JSON Preparation Parser
    json_parser = subparsers.add_parser('prepare_json', help='Prepare JSON datasets')
    json_parser.add_argument("-i", "--input_dir", type=Path, required=True,
                          help="Directory containing processed JSON files")
    json_parser.add_argument("-o", "--output_dir", type=Path, required=True,
                          help="Output directory for training data")
    json_parser.add_argument("-j", "--n_jobs", type=int, default=1,
                          help="Parallel processing jobs")
    json_parser.add_argument("-q", "--quiet", action="store_true",
                          help="Reduce output verbosity")
    
    return parser.parse_args()

def sanitize_midi_value(value, min_val=0, max_val=127):
    """Clamp MIDI values to valid range with type conversion."""
    try:
        return min(max(int(float(value)), min_val), max_val)
    except (ValueError, TypeError):
        return min_val

def try_read_midi(filename, attempts=3):
    """Robust MIDI reading with multiple fallback strategies."""
    for attempt in range(attempts):
        try:
            # Attempt 1: Lenient muspy reading
            if attempt == 0:
                return muspy.read_midi(filename, strict=False)
            
            # Attempt 2: Mido with sanitization
            elif attempt == 1:
                midi = MidoFile(filename, clip=True)
                return muspy.from_mido(midi)
            
            # Attempt 3: Manual track reconstruction
            elif attempt == 2:
                midi = MidiFile(filename, clip=True)
                sanitized = MidiFile(ticks_per_beat=midi.ticks_per_beat or 480)
                
                for track in midi.tracks:
                    new_track = MidiTrack()
                    valid_notes = 0
                    
                    for msg in track:
                        try:
                            if msg.type in ['note_on', 'note_off', 'set_tempo']:
                                if msg.type in ['note_on', 'note_off']:
                                    msg.velocity = sanitize_midi_value(getattr(msg, 'velocity', 64))
                                    msg.note = sanitize_midi_value(getattr(msg, 'note', 60))
                                new_track.append(msg.copy())
                                valid_notes += 1
                        except Exception:
                            continue
                    
                    if valid_notes > 0:
                        sanitized.tracks.append(new_track)
                
                if len(sanitized.tracks) > 0:
                    return muspy.from_mido(sanitized)
        
        except Exception as e:
            logging.debug(f"Attempt {attempt+1} failed for {filename}:\n{traceback.format_exc()}")
            continue
    
    raise ValueError(f"All {attempts} reading attempts failed")

def get_lmd_instrument(program, is_drum):
    """Enhanced instrument mapping with comprehensive ranges."""
    if is_drum:
        return "Drums"
    
    program = sanitize_midi_value(program)
    
    instrument_map = [
        (0, 8, "Piano"),        # Piano
        (24, 32, "Guitar"),     # Guitar
        (32, 40, "Bass"),       # Bass
        (40, 48, "Strings"),    # Strings
        (48, 56, "Strings"),    # More Strings
        (56, 64, "Brass"),     # Brass
        (64, 72, "Reed"),       # Reed
        (72, 80, "Pipe"),      # Pipe
        (80, 104, "Synth"),    # Synth
        (104, 112, "Ethnic"),  # Ethnic
        (112, 120, "Percussion") # Percussion
    ]
    
    for start, end, name in instrument_map:
        if start <= program < end:
            return name
    return "Piano"  # Default fallback

def process_midi_file(filename, args):
    """Comprehensive MIDI processing with enhanced validation."""
    try:
        # Read MIDI with appropriate strategy
        music = try_read_midi(filename) if not args.skip_errors else muspy.read_midi(filename, strict=False)
        
        if not music or not music.tracks:
            logging.debug(f"Empty MIDI file: {filename}")
            return None

        # Standardize resolution
        try:
            music.adjust_resolution(LMD_CONFIG["resolution"])
        except Exception:
            music.resolution = LMD_CONFIG["resolution"]
        
        # Duration filtering
        total_duration = music.get_real_end_time()
        if total_duration > LMD_CONFIG["max_duration"] or total_duration < 10:
            logging.debug(f"Invalid duration ({total_duration}) in {filename}")
            return None

        # Process tracks and notes
        notes_dict = defaultdict(list)
        for track in music.tracks:
            instrument = get_lmd_instrument(track.program, track.is_drum)
            if not instrument:
                continue

            for note in track.notes:
                try:
                    notes_dict[instrument].append({
                        "time": note.time,
                        "pitch": sanitize_midi_value(note.pitch),
                        "duration": min(note.duration, LMD_CONFIG["max_duration_ticks"]),
                        "velocity": sanitize_midi_value(getattr(note, 'velocity', 64))
                    })
                except Exception:
                    continue

        # Filter tracks by note count
        notes_dict = {
            k: v for k, v in notes_dict.items()
            if len(v) >= args.min_notes or k == "Drums"
        }
        
        # Require at least 2 non-drum instruments
        if sum(1 for k in notes_dict if k != "Drums") < 2:
            logging.debug(f"Insufficient instruments in {filename}")
            return None

        # Prepare output structure
        return {
            "metadata": {
                "title": filename.stem,
                "source": "LMD",
                "duration": total_duration,
                "parts": list(notes_dict.keys())
            },
            "tracks": [
                {
                    "name": inst,
                    "program": LMD_CONFIG["programs"].get(inst, 0),
                    "is_drum": (inst == "Drums"),
                    "notes": sorted(notes, key=lambda x: (x["time"], x["pitch"]))
                }
                for inst, notes in notes_dict.items()
            ]
        }

    except Exception as e:
        logging.error(f"❌ Processing failed for {filename}:\n{traceback.format_exc()}")
        failed_files.append(str(filename))
        return None

def find_midi_files(input_dir):
    """Recursively find MIDI files in nested directories."""
    midi_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                midi_files.append(Path(root) / file)
    return midi_files

def save_results(output_dir, json_data):
    """Save processed data with proper directory structure."""
    random.shuffle(json_data)
    split_ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}
    
    # Create split directories
    for split_name in split_ratios:
        (output_dir / split_name).mkdir(exist_ok=True, parents=True)
    
    # Split data
    splits = {}
    start = 0
    for name, ratio in split_ratios.items():
        end = start + int(ratio * len(json_data))
        splits[name] = json_data[start:end]
        start = end
    
    # Save each split
    for split_name, data in splits.items():
        split_dir = output_dir / split_name
        logging.info(f"Saving {len(data)} files to {split_name} set")
        
        # Save individual files
        for name, content in data:
            output_path = split_dir / f"{name}.json.gz"
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False)
        
        # Save filename index
        with open(split_dir / "filenames.txt", 'w', encoding='utf-8') as f:
            f.writelines(f"{n}\n" for n, _ in data)

def process_midi_files(args):
    """Main processing pipeline with resume capability."""
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Locate MIDI files
    logging.info(f"Scanning {args.input_dir} for MIDI files...")
    midi_files = find_midi_files(args.input_dir)
    
    if not midi_files:
        logging.error("No MIDI files found in input directory!")
        return
    
    # Apply sample limit if specified
    if args.max_samples:
        midi_files = random.sample(midi_files, min(args.max_samples, len(midi_files)))
    
    logging.info(f"Found {len(midi_files)} files to process")
    
    # Filter already processed files
    processed_files = set()
    processed_log = args.output_dir / "processed_files.txt"
    if processed_log.exists():
        with open(processed_log, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f)
    
    midi_files = [f for f in midi_files if f.name not in processed_files]
    logging.info(f"Processing {len(midi_files)} new files")
    
    # Parallel processing
    if args.n_jobs > 1:
        import multiprocessing
        args.n_jobs = min(args.n_jobs, multiprocessing.cpu_count() - 1)
        results = joblib.Parallel(n_jobs=args.n_jobs, verbose=0)(
            joblib.delayed(lambda f: (f.stem, process_midi_file(f, args)))(f) 
            for f in tqdm.tqdm(midi_files, desc="Processing MIDI files")
        )
    else:
        results = []
        for f in tqdm.tqdm(midi_files, desc="Processing MIDI files"):
            results.append((f.stem, process_midi_file(f, args)))
    
    # Filter successful results
    valid_results = [(name, data) for name, data in results if data]
    logging.info(f"Success rate: {len(valid_results)}/{len(midi_files)} ({len(valid_results)/len(midi_files):.1%})")
    
    # Save results
    if valid_results:
        save_results(args.output_dir, valid_results)
        
        # Update processed files log
        with open(processed_log, 'a', encoding='utf-8') as f:
            f.writelines(f"{name}\n" for name, _ in valid_results)
    
    # Save failure log
    if failed_files:
        with open(args.output_dir / "failed_files.txt", 'w', encoding='utf-8') as f:
            f.writelines(f"{fn}\n" for fn in failed_files)
        logging.warning(f"{len(failed_files)} files failed (see failed_files.txt)")

def prepare_json_files(args):
    """Prepare JSON datasets for training (placeholder)."""
    args.output_dir.mkdir(exist_ok=True, parents=True)
    logging.info("JSON preparation functionality not yet implemented")

def main():
    """Main entry point with error handling."""
    try:
        args = parse_arguments()
        setup_logging(args.output_dir if hasattr(args, 'output_dir') else None, args.quiet)
        
        if args.command == "process_midi":
            process_midi_files(args)
        elif args.command == "prepare_json":
            prepare_json_files(args)
            
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        exit(1)

if __name__ == "__main__":
    main()