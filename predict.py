"""
Unified Prediction Script for LSTM, Transformer, and Hybrid Models.
Supports MIDI input and outputs multi-track MIDI with instrument assignments.
"""

import argparse
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import muspy
from tqdm import tqdm

# ===== Configuration =====
CONFIG = {
    "bach": {
        "programs": {
            "Soprano": 73,
            "Alto": 68,
            "Tenor": 70,
            "Bass": 71
        }
    },
    "lmd": {
        "programs": {
            "Piano": 0,
            "Guitar": 24,
            "Bass": 32,
            "Strings": 48,
            "Brass": 61
        }
    }
}

# ===== Setup Logging =====
def setup_logger(quiet=False):
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')

# ===== Import Model Definitions =====
# Note: تأكدي أن ملفات النماذج موجودة في مجلد models/
# models/lstm.py يحتوي على class LSTMArranger
# models/transformer.py يحتوي على class TransformerArranger
# models/hybrid.py يحتوي على def create_hybrid_model

def get_model_class(model_type):
    """Return the appropriate model class/function based on type."""
    if model_type == "lstm":
        from models.lstm import LSTMArranger
        return LSTMArranger, None
    elif model_type == "transformer":
        from models.transformer import TransformerArranger
        return TransformerArranger, None
    elif model_type == "hybrid":
        from models.hybrid import create_hybrid_model
        return None, create_hybrid_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ===== MIDI Processing =====
def load_midi_with_duration(midi_path, max_beat=16384):
    """
    Load MIDI and extract notes with their actual durations.
    Returns: list of dicts with time, pitch, duration, velocity.
    """
    try:
        music = muspy.read_midi(midi_path)
        notes = []
        
        for track in music.tracks:
            if track.is_drum:
                continue
            for note in track.notes:
                # Clip time to max_beat
                time_val = min(note.time, max_beat - 1)
                # Duration is already in ticks/beats from muspy
                duration_val = min(note.duration, max_beat - time_val)
                notes.append({
                    "time": time_val,
                    "pitch": note.pitch,
                    "duration": duration_val,
                    "velocity": note.velocity if hasattr(note, 'velocity') else 64,
                    "original_track": f"Track_{track.program}" if hasattr(track, 'program') else "Unknown"
                })
        
        # Sort by time
        notes.sort(key=lambda x: x["time"])
        return notes, music.resolution if hasattr(music, 'resolution') else 480
    except Exception as e:
        logging.error(f"Error loading MIDI: {e}")
        return None, None

def prepare_sequences(notes, seq_len):
    """Split notes into overlapping sequences for prediction."""
    if len(notes) < seq_len:
        # Pad if too short
        padded = notes + [notes[-1]] * (seq_len - len(notes))
        return [padded]
    
    sequences = []
    # Use sliding window with step = seq_len // 2 for smooth predictions
    step = seq_len // 2
    for i in range(0, len(notes) - seq_len + 1, step):
        sequences.append(notes[i:i+seq_len])
    
    # Ensure last sequence is included
    if (len(notes) - seq_len) % step != 0:
        sequences.append(notes[-seq_len:])
    
    return sequences

def predict_notes(model, sequences, model_type, use_duration=False):
    """Run prediction on all sequences and return track labels."""
    all_predictions = []
    
    for seq in sequences:
        time_seq = np.array([n["time"] for n in seq], dtype=np.int32)
        pitch_seq = np.array([n["pitch"] for n in seq], dtype=np.int32)
        
        # Prepare inputs based on model type
        if model_type == "hybrid" or model_type == "lstm":
            # These models expect [time_seq, pitch_seq] as separate inputs
            inputs = [np.expand_dims(time_seq, 0), np.expand_dims(pitch_seq, 0)]
        else:  # transformer
            # Transformer expects dict
            inputs = {
                "time": np.expand_dims(time_seq, 0),
                "pitch": np.expand_dims(pitch_seq, 0)
            }
            if use_duration:
                dur_seq = np.array([n["duration"] for n in seq], dtype=np.int32)
                inputs["duration"] = np.expand_dims(dur_seq, 0)
        
        # Predict
        pred = model.predict(inputs, verbose=0)
        # Get argmax (track index)
        if isinstance(pred, list):
            pred = pred[0]
        track_indices = np.argmax(pred, axis=-1).flatten()
        all_predictions.extend(track_indices.tolist())
    
    return all_predictions

# ===== Reconstruct MIDI =====
def reconstruct_midi(original_notes, predictions, dataset_config, ticks_per_beat):
    """Create a multi-track MIDI from predictions."""
    track_names = list(dataset_config["programs"].keys())
    program_map = dataset_config["programs"]
    
    # Create a new MusPy Music object
    music = muspy.Music(resolution=ticks_per_beat)
    
    # Create a track for each instrument
    tracks = {}
    for name in track_names:
        track = muspy.Track(program=program_map[name], is_drum=False)
        track.name = name
        tracks[name] = track
    
    # Assign notes to tracks
    # Ensure predictions length matches notes length
    min_len = min(len(original_notes), len(predictions))
    
    for i in range(min_len):
        note_data = original_notes[i]
        track_idx = predictions[i]
        
        # Ensure track_idx is valid
        if track_idx >= len(track_names):
            track_idx = 0
        
        track_name = track_names[track_idx]
        
        # Create note
        new_note = muspy.Note(
            time=note_data["time"],
            pitch=note_data["pitch"],
            duration=note_data["duration"],
            velocity=note_data.get("velocity", 64)
        )
        tracks[track_name].notes.append(new_note)
    
    # Add tracks to music (skip empty tracks)
    for name, track in tracks.items():
        if track.notes:
            music.tracks.append(track)
    
    return music

# ===== Save JSON Report =====
def save_json_report(original_notes, predictions, track_names, output_path):
    """Save detailed prediction report."""
    report = {
        "total_notes": len(predictions),
        "track_distribution": {},
        "predictions": []
    }
    
    for i, name in enumerate(track_names):
        count = predictions.count(i)
        report["track_distribution"][name] = count
    
    for i, (note, pred_idx) in enumerate(zip(original_notes, predictions)):
        report["predictions"].append({
            "index": i,
            "time": int(note["time"]),
            "pitch": int(note["pitch"]),
            "duration": int(note["duration"]),
            "predicted_track": track_names[pred_idx] if pred_idx < len(track_names) else "Unknown"
        })
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to: {output_path}")

# ===== Main Function =====
def main():
    parser = argparse.ArgumentParser(description="Unified Music Arrangement Predictor")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                       help="Path to trained model (.keras file)")
    parser.add_argument("-i", "--input_midi", type=str, required=True,
                       help="Input MIDI file path")
    parser.add_argument("-o", "--output_midi", type=str, required=True,
                       help="Output MIDI file path")
    parser.add_argument("-d", "--dataset", required=True, choices=("bach", "lmd"),
                       help="Dataset key (bach or lmd)")
    parser.add_argument("-t", "--model_type", required=True, choices=("lstm", "transformer", "hybrid"),
                       help="Model type")
    parser.add_argument("--use_duration", action="store_true",
                       help="Use duration as input (if model was trained with -di)")
    parser.add_argument("--seq_len", type=int, default=100,
                       help="Sequence length for prediction (default: 100)")
    parser.add_argument("--max_beat", type=int, default=16384,
                       help="Maximum beat value")
    parser.add_argument("--json_output", type=str, default="predictions.json",
                       help="Path to save JSON report")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    setup_logger(args.quiet)
    
    # ===== 1. Load Model =====
    logging.info(f"Loading {args.model_type} model from: {args.model_path}")
    
    # Get the model class/function
    model_class, model_fn = get_model_class(args.model_type)
    
    # Load with custom objects
    custom_objects = {}
    if args.model_type == "lstm":
        custom_objects["LSTMArranger"] = model_class
    elif args.model_type == "transformer":
        custom_objects["TransformerArranger"] = model_class
    # Hybrid doesn't have a custom class, it's a functional model
    
    try:
        if args.model_type == "hybrid":
            # Hybrid model uses functional API, load directly
            model = tf.keras.models.load_model(args.model_path)
        else:
            model = tf.keras.models.load_model(
                args.model_path,
                custom_objects=custom_objects
            )
        logging.info("Model loaded successfully!")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.error("Make sure the model file exists and matches the model type.")
        return
    
    # ===== 2. Load MIDI =====
    logging.info(f"Loading MIDI: {args.input_midi}")
    notes, ticks_per_beat = load_midi_with_duration(args.input_midi, args.max_beat)
    if notes is None:
        return
    
    logging.info(f"Extracted {len(notes)} notes from MIDI")
    
    # ===== 3. Prepare Sequences =====
    sequences = prepare_sequences(notes, args.seq_len)
    logging.info(f"Created {len(sequences)} sequences for prediction")
    
    # ===== 4. Predict =====
    logging.info("Running prediction...")
    predictions = predict_notes(model, sequences, args.model_type, args.use_duration)
    
    # If we have more predictions than notes (due to overlapping windows), trim
    if len(predictions) > len(notes):
        predictions = predictions[:len(notes)]
    elif len(predictions) < len(notes):
        # Pad with first track (0)
        predictions.extend([0] * (len(notes) - len(predictions)))
    
    logging.info(f"Generated {len(predictions)} track assignments")
    
    # ===== 5. Reconstruct MIDI =====
    dataset_config = CONFIG[args.dataset]
    track_names = list(dataset_config["programs"].keys())
    
    logging.info("Reconstructing multi-track MIDI...")
    music = reconstruct_midi(notes, predictions, dataset_config, ticks_per_beat)
    
    # Save MIDI
    muspy.write_midi(args.output_midi, music)
    logging.info(f"Output MIDI saved to: {args.output_midi}")
    
    # ===== 6. Save JSON Report =====
    save_json_report(notes, predictions, track_names, args.json_output)
    
    # ===== 7. Display Summary =====
    print("\n" + "="*50)
    print("🎵 PREDICTION SUMMARY")
    print("="*50)
    print(f"Total notes processed: {len(predictions)}")
    print("\nTrack Distribution:")
    for i, name in enumerate(track_names):
        count = predictions.count(i)
        pct = (count / len(predictions)) * 100 if predictions else 0
        print(f"  {name}: {count} notes ({pct:.2f}%)")
    print("="*50)
    logging.info("Done!")

if __name__ == "__main__":
    main()