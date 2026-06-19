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
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')

# ===== Parse Arguments =====
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="input data directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("-d", "--dataset", required=True, choices=("bach", "lmd"), help="dataset key")
    parser.add_argument("-sl", "--seq_len", type=int, default=50, help="sequence length")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-e", "--epoch", type=int, default=50, help="number of epochs")
    parser.add_argument("-q", "--quiet", action="store_true", help="reduce output verbosity")
    parser.add_argument("-mb", "--max_beat", type=int, default=16384, help="maximum beat value")
    return parser.parse_args()

# ===== Data Processing =====
def load_and_process_data(data_dir, split, dataset_config, seq_len, max_beat):
    data_path = Path(data_dir) / split
    files = list(data_path.glob("*.json"))
    if not files:
        logging.warning(f"No JSON files found in: {data_path}")
        return [], []
    
    valid_track_names = list(dataset_config["programs"].keys())
    sequences, labels = [], []
    
    for file in files:
        try:
            with open(file, 'r') as f:
                piece = json.load(f)
                for track in piece["tracks"]:
                    if track["name"] not in valid_track_names:
                        continue
                    label_idx = valid_track_names.index(track["name"])
                    note_sequence = []
                    for note in track["notes"]:
                        time_key = "offset" if "offset" in note else "time"
                        time_value = min(note.get(time_key, 0), max_beat - 1)
                        note_sequence.append([time_value, note["pitch"]])
                    
                    for i in range(0, len(note_sequence) - seq_len + 1, seq_len):
                        seq = note_sequence[i:i+seq_len]
                        sequences.append(seq)
                        labels.append([label_idx] * seq_len)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            continue
    return sequences, labels

# ===== Hybrid Model (LSTM + Transformer) =====
def create_hybrid_model(seq_len, n_tracks, max_beat):
    time_input = tf.keras.Input(shape=(seq_len,), name="time_input")
    pitch_input = tf.keras.Input(shape=(seq_len,), name="pitch_input")

    # Embeddings
    time_emb = tf.keras.layers.Embedding(max_beat, 32)(time_input)
    pitch_emb = tf.keras.layers.Embedding(128, 32)(pitch_input)
    combined = tf.keras.layers.Concatenate()([time_emb, pitch_emb])

    # ---- Branch 1: LSTM ----
    lstm_branch = tf.keras.layers.LSTM(64, return_sequences=True)(combined)
    lstm_branch = tf.keras.layers.Dropout(0.2)(lstm_branch)

    # ---- Branch 2: Transformer ----
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(combined, combined)
    x = tf.keras.layers.Add()([combined, attn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64)
    ])
    transformer_branch = tf.keras.layers.Add()([x, ffn(x)])
    transformer_branch = tf.keras.layers.LayerNormalization()(transformer_branch)

    # ---- Merge Branches ----
    merged = tf.keras.layers.Concatenate()([lstm_branch, transformer_branch])
    merged = tf.keras.layers.Dense(128, activation="relu")(merged)
    merged = tf.keras.layers.Dropout(0.3)(merged)

    # Output
    outputs = tf.keras.layers.Dense(n_tracks, activation="softmax")(merged)

    return tf.keras.Model(inputs=[time_input, pitch_input], outputs=outputs)

# ===== Main Function =====
def main():
    args = parse_arguments()
    setup_logger(args.quiet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_config = CONFIG[args.dataset]
    n_tracks = len(dataset_config["programs"])

    logging.info("Loading and processing data...")
    train_sequences, train_labels = load_and_process_data(args.input_dir, "train", dataset_config, args.seq_len, args.max_beat)
    val_sequences, val_labels = load_and_process_data(args.input_dir, "valid", dataset_config, args.seq_len, args.max_beat)

    if not train_sequences:
        logging.error("No training data found!")
        return

    train_sequences, train_labels = np.array(train_sequences), np.array(train_labels)
    val_sequences, val_labels = np.array(val_sequences), np.array(val_labels)

    # Clip ranges
    train_sequences[:, :, 0] = np.clip(train_sequences[:, :, 0], 0, args.max_beat - 1)
    train_sequences[:, :, 1] = np.clip(train_sequences[:, :, 1], 0, 127)
    val_sequences[:, :, 0] = np.clip(val_sequences[:, :, 0], 0, args.max_beat - 1)
    val_sequences[:, :, 1] = np.clip(val_sequences[:, :, 1], 0, 127)

    # Split
    train_time, train_pitch = train_sequences[:, :, 0], train_sequences[:, :, 1]
    val_time, val_pitch = val_sequences[:, :, 0], val_sequences[:, :, 1]

    logging.info("Creating hybrid model...")
    model = create_hybrid_model(args.seq_len, n_tracks, args.max_beat)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Added EarlyStopping only
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    logging.info("Training hybrid model...")
    history = model.fit(
        [train_time, train_pitch],
        train_labels,
        validation_data=([val_time, val_pitch], val_labels),
        epochs=args.epoch,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1 if not args.quiet else 0
    )

    model.save(output_dir / "hybrid_model.keras")
    logging.info("Training completed!")

if __name__ == "__main__":
    main()
