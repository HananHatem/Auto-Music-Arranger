"""Train the Music Arrangement Model with Simplified Transformer Encoder."""
import os
import argparse
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

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
    """Setup logger."""
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')

# ===== Parse Arguments =====
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="input data directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("-d", "--dataset", required=True, choices=("bach", "lmd"), help="dataset key")
    parser.add_argument("-sl", "--seq_len", type=int, default=100, help="sequence length")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-e", "--epoch", type=int, default=50, help="number of epochs")
    parser.add_argument("-q", "--quiet", action="store_true", help="reduce output verbosity")
    parser.add_argument("-mb", "--max_beat", type=int, default=16384, help="maximum beat value")
    return parser.parse_args()

# ===== Data Processing =====
def load_and_process_data(data_dir, split, dataset_config, seq_len, max_beat):
    """Load and process JSON data into sequences."""
    data_path = Path(data_dir) / split
    files = list(data_path.glob("*.json"))
    
    if not files:
        logging.warning(f"No JSON files found in: {data_path}")
        return [], []
    
    valid_track_names = list(dataset_config["programs"].keys())
    sequences = []
    labels = []
    
    for file in files:
        try:
            with open(file, 'r') as f:
                piece = json.load(f)
                
                # Process each track
                for track in piece["tracks"]:
                    if track["name"] not in valid_track_names:
                        continue
                        
                    label_idx = valid_track_names.index(track["name"])
                    note_sequence = []
                    
                    for note in track["notes"]:
                        time_key = "offset" if "offset" in note else "time"
                        time_value = note.get(time_key, 0)
                        
                        # Clip time values to be within the allowed range
                        time_value = min(time_value, max_beat - 1)
                        
                        note_data = [
                            time_value,
                            note["pitch"]
                        ]
                        note_sequence.append(note_data)
                    
                    # Split into sequences of fixed length
                    for i in range(0, len(note_sequence) - seq_len + 1, seq_len):
                        seq = note_sequence[i:i+seq_len]
                        sequences.append(seq)
                        labels.append([label_idx] * seq_len)
                        
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            continue
    
    return sequences, labels

# ===== Simplified Transformer Model =====
def create_simple_transformer_model(seq_len, n_tracks, max_beat):
    """Create a simplified transformer-based model."""
    # Input layers
    time_input = tf.keras.Input(shape=(seq_len,), name="time_input")
    pitch_input = tf.keras.Input(shape=(seq_len,), name="pitch_input")
    
    # Simple embeddings with appropriate vocabulary sizes
    time_embedding = tf.keras.layers.Embedding(max_beat, 32)(time_input)
    pitch_embedding = tf.keras.layers.Embedding(128, 32)(pitch_input)
    
    # Combine embeddings
    combined = tf.keras.layers.Concatenate()([time_embedding, pitch_embedding])
    
    # Simple transformer layer using MultiHeadAttention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=64
    )(combined, combined)
    
    # Add & Norm
    x = tf.keras.layers.Add()([combined, attention_output])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Feed forward
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64)
    ])
    ffn_output = ffn(x)
    
    # Add & Norm
    x = tf.keras.layers.Add()([x, ffn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(n_tracks, activation='softmax')(x)
    
    return tf.keras.Model(inputs=[time_input, pitch_input], outputs=outputs)

# ===== Main Function =====
def main():
    """Main function."""
    args = parse_arguments()
    setup_logger(args.quiet)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    dataset_config = CONFIG[args.dataset]
    n_tracks = len(dataset_config["programs"])
    
    logging.info("Loading and processing data...")
    train_sequences, train_labels = load_and_process_data(
        args.input_dir, "train", dataset_config, args.seq_len, args.max_beat
    )
    val_sequences, val_labels = load_and_process_data(
        args.input_dir, "valid", dataset_config, args.seq_len, args.max_beat
    )
    
    if not train_sequences:
        logging.error("No training data found!")
        return
    
    # Convert to numpy arrays
    train_sequences = np.array(train_sequences)
    train_labels = np.array(train_labels)
    val_sequences = np.array(val_sequences)
    val_labels = np.array(val_labels)
    
    # Verify data ranges
    max_time = np.max(train_sequences[:, :, 0])
    max_pitch = np.max(train_sequences[:, :, 1])
    logging.info(f"Max time value: {max_time}, Max pitch value: {max_pitch}")
    
    # Clip values to ensure they're within expected ranges
    train_sequences[:, :, 0] = np.clip(train_sequences[:, :, 0], 0, args.max_beat - 1)
    train_sequences[:, :, 1] = np.clip(train_sequences[:, :, 1], 0, 127)
    val_sequences[:, :, 0] = np.clip(val_sequences[:, :, 0], 0, args.max_beat - 1)
    val_sequences[:, :, 1] = np.clip(val_sequences[:, :, 1], 0, 127)
    
    # Split sequences into time and pitch components
    train_time = train_sequences[:, :, 0]
    train_pitch = train_sequences[:, :, 1]
    val_time = val_sequences[:, :, 0]
    val_pitch = val_sequences[:, :, 1]
    
    # Create model
    logging.info("Creating simplified transformer model...")
    model = create_simple_transformer_model(args.seq_len, n_tracks, args.max_beat)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary(print_fn=logging.info)
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
    ]
    
    # Train model
    logging.info("Training model...")
    history = model.fit(
        [train_time, train_pitch],
        train_labels,
        validation_data=([val_time, val_pitch], val_labels),
        epochs=args.epoch,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1 if not args.quiet else 0
    )
    
    # Save model
    model.save(output_dir / "transformer_model.keras")
    logging.info("Training completed!")

if __name__ == "__main__":
    main()