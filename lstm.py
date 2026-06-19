"""Train the Music Arrangement Model."""
import os
import sys
import argparse
import json
import logging
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

# ===== إعدادات البيانات =====
CONFIG = {
    "bach": {
        "programs": {
            "Soprano": 73,  # flute
            "Alto": 68,      # oboe
            "Tenor": 70,     # bassoon
            "Bass": 71       # clarinet
        },
        "has_drums": False
    },
    "lmd": {
        "programs": {
            "Piano": 0,      # acoustic grand piano
            "Guitar": 24,    # acoustic guitar (nylon)
            "Bass": 32,      # acoustic bass
            "Strings": 48,   # string ensemble 1
            "Brass": 61      # brass section
        },
        "has_drums": True
    },
    "colors": [
        [31, 119, 180],    # blue
        [255, 127, 14],    # orange
        [44, 160, 44],     # green
        [214, 39, 40],     # red
        [148, 103, 189],   # purple
        [140, 86, 75]      # brown
    ]
}

# ===== إعداد السجلات =====
def setup_loggers(filename=None, quiet=False):
    """Setup loggers."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if not quiet else logging.ERROR)
    
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

# ===== معالجة وسائط سطر الأوامر =====
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="input data directory")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="output directory")
    parser.add_argument("-d", "--dataset", required=True, choices=("bach", "lmd"), help="dataset key")
    parser.add_argument("-nau", "--no_augmentation", dest="augmentation", action="store_false", help="whether to use data augmentation")
    parser.set_defaults(augmentation=True)
    parser.add_argument("-sl", "--seq_len", type=int, default=500, help="sequence length")
    parser.add_argument("-ml", "--max_len", type=int, default=2000, help="maximum sequence length for validation")
    parser.add_argument("-di", "--use_duration", action="store_true", help="use duration as an input")
    parser.add_argument("-fi", "--use_frequency", action="store_true", help="use frequency as an input")
    parser.add_argument("-oh", "--use_onset_hint", action="store_true", help="use onset hint as an input")
    parser.add_argument("-ph", "--use_pitch_hint", action="store_true", help="use pitch hint as an input")
    parser.add_argument("-pe", "--use_pitch_embedding", action="store_true", help="use pitch embedding")
    parser.add_argument("-te", "--use_time_embedding", action="store_true", help="use time embedding")
    parser.add_argument("-de", "--use_duration_embedding", action="store_true", help="use duration embedding")
    parser.add_argument("-mb", "--max_beat", type=int, default=4096, help="maximum number of beats")
    parser.add_argument("-md", "--max_duration", type=int, default=192, help="maximum duration")
    parser.add_argument("-nl", "--n_layers", type=int, default=2, help="number of LSTM layers")
    parser.add_argument("-dm", "--d_model", type=int, default=128, help="number of hidden units for the LSTM layer")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="maximum number of epochs")
    parser.add_argument("-s", "--steps_per_epoch", type=int, default=500, help="number of steps per epochs")
    parser.add_argument("-p", "--patience", type=int, default=5, help="patience for early stopping")
    parser.add_argument("-g", "--gpu", type=int, help="GPU device to use")
    parser.add_argument("-q", "--quiet", action="store_true", help="reduce output verbosity")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    return parser.parse_args()

# ===== وظائف معالجة البيانات =====
def load_json_data(data_dir, split, max_beat, dataset_config):
    """Load JSON data from directory with proper track filtering."""
    data = {
        "time": [],
        "pitch": [],
        "duration": [],
        "label": []
    }
    
    files = list((data_dir / split).glob("*.json"))
    if not files:
        logging.warning(f"No JSON files found in directory: {data_dir / split}")
        return data
    
    valid_track_names = list(dataset_config["programs"].keys())
    
    for file in files:
        try:
            with open(file, 'r') as f:
                piece = json.load(f)
                
                # تحديد أسماء المسارات الصالحة فقط
                track_names = piece["metadata"]["parts"]
                valid_indices = [i for i, name in enumerate(track_names) if name in valid_track_names]
                
                if not valid_indices:
                    continue
                
                # إنشاء خريطة التسميات
                track_map = {name: idx for idx, name in enumerate(valid_track_names)}
                
                # جمع جميع النوتات من المسارات الصالحة فقط
                all_notes = []
                for track_idx, track in enumerate(piece["tracks"]):
                    if track_idx not in valid_indices:
                        continue
                    track_name = track["name"]
                    if track_name not in track_map:
                        continue
                    label = track_map[track_name]
                    for note in track["notes"]:
                        # تحديد مفتاح الوقت (يدعم كل من 'offset' و 'time')
                        time_key = "offset" if "offset" in note else "time"
                        time_value = note.get(time_key, 0)
                        
                        # تطبيق الحدود على قيمة الوقت
                        time_value = max(0, min(time_value, max_beat - 1))
                        
                        all_notes.append({
                            "time": time_value,
                            "pitch": note["pitch"],
                            "duration": note["duration"],
                            "label": label
                        })
                
                # تخطي إذا لم تكن هناك نوتات صالحة
                if not all_notes:
                    continue
                    
                # ترتيب النوتات حسب الوقت
                all_notes.sort(key=lambda x: x["time"])
                
                # إضافة النوتات إلى البيانات
                data["time"].append(np.array([note["time"] for note in all_notes], dtype=np.int32))
                data["pitch"].append(np.array([note["pitch"] for note in all_notes], dtype=np.int32))
                data["duration"].append(np.array([note["duration"] for note in all_notes], dtype=np.int32))
                data["label"].append(np.array([note["label"] for note in all_notes], dtype=np.int32))
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            continue
    
    return data

def analyze_data(data, n_tracks, name):
    """Analyze data distribution and check for issues."""
    logging.info(f"Analyzing {name} data...")
    
    total_sequences = len(data["label"])
    total_notes = sum(len(seq) for seq in data["label"])
    
    # Check label distribution
    label_counts = np.zeros(n_tracks, dtype=np.int32)
    for labels in data["label"]:
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            if 0 <= u < n_tracks:
                label_counts[u] += c
    
    logging.info(f"Total sequences: {total_sequences}")
    logging.info(f"Total notes: {total_notes}")
    logging.info(f"Label distribution: {label_counts}")
    
    # Check for data leakage
    leakage_count = 0
    for i in range(len(data["label"])):
        pitch = data["pitch"][i]
        time = data["time"][i]
        labels = data["label"][i]
        
        # Check if labels match input features
        if np.any(pitch == labels) or np.any(time == labels):
            leakage_count += 1
    
    if leakage_count > 0:
        logging.warning(f"Data leakage detected in {leakage_count}/{total_sequences} sequences!")

def create_dataset(data, n_tracks, args, training=False):
    """Create reliable dataset without data leakage."""
    sequences = []
    labels = []
    
    for i in range(len(data["label"])):
        # Skip empty sequences
        if len(data["time"][i]) == 0:
            continue
            
        # Get sequence
        time_seq = data["time"][i]
        pitch_seq = data["pitch"][i]
        label_seq = data["label"][i]
        
        if args.use_duration:
            dur_seq = data["duration"][i]
        
        # Apply truncation if sequence is too long
        if len(time_seq) > args.max_len:
            time_seq = time_seq[:args.max_len]
            pitch_seq = pitch_seq[:args.max_len]
            label_seq = label_seq[:args.max_len]
            if args.use_duration:
                dur_seq = dur_seq[:args.max_len]
        
        # Apply clipping
        time_seq = np.clip(time_seq, 0, args.max_beat - 1)
        pitch_seq = np.clip(pitch_seq, 0, 127)
        
        if args.use_duration:
            dur_seq = np.clip(dur_seq, 0, args.max_duration - 1)
        
        # Skip sequences with invalid labels
        if np.any((label_seq < 0) | (label_seq >= n_tracks)):
            continue
            
        # Skip sequences with data leakage
        if np.any(pitch_seq == label_seq) or np.any(time_seq == label_seq):
            continue
            
        # Apply augmentation if training
        if training and args.augmentation and len(pitch_seq) > 10:
            # Random transpose
            pitch_shift = random.randint(-5, 6)
            pitch_seq = pitch_seq + pitch_shift
            pitch_seq[pitch_seq > 127] -= 12
            pitch_seq[pitch_seq < 0] += 12
            pitch_seq = np.clip(pitch_seq, 0, 127)
        
        # Create input dictionary
        inputs = {"time": time_seq, "pitch": pitch_seq}
        if args.use_duration:
            inputs["duration"] = dur_seq
        
        sequences.append(inputs)
        labels.append(label_seq)
    
    # Create TensorFlow dataset
    def gen():
        for i in range(len(sequences)):
            yield sequences[i], labels[i]
    
    output_signature = (
        {
            "time": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "pitch": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    
    if args.use_duration:
        output_signature[0]["duration"] = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )            

# ===== تعريفات النموذج =====
class LSTMArranger(tf.keras.Model):
    """LSTM-based music arrangement model - Simplified version"""
    
    def __init__(self, use_duration, n_tracks, d_model, n_layers, **kwargs):
        super().__init__()
        self.use_duration = use_duration
        self.n_tracks = n_tracks
        
        # Input embeddings
        self.pitch_emb = tf.keras.layers.Embedding(128, 64, mask_zero=True)
        self.time_emb = tf.keras.layers.Embedding(4096, 64, mask_zero=True)
        
        if use_duration:
            self.dur_emb = tf.keras.layers.Embedding(192, 32, mask_zero=True)
        
        # LSTM layers
        self.lstm_layers = []
        for _ in range(n_layers):
            self.lstm_layers.append(tf.keras.layers.LSTM(
                d_model, 
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1
            ))
        
        # Solfege-inspired output layer
        self.solfegge_layer = tf.keras.layers.Dense(128, activation='relu')
        self.time_distributed = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(n_tracks, activation='softmax')
            ])
        )
    
    def call(self, inputs):
        pitch_emb = self.pitch_emb(inputs["pitch"])
        time_emb = self.time_emb(inputs["time"])
        x = tf.concat([pitch_emb, time_emb], axis=-1)

        if self.use_duration:
            dur_emb = self.dur_emb(inputs["duration"])
            x = tf.concat([x, dur_emb], axis=-1)

        # Process through LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        
        # Apply solfege-inspired transformation
        x = self.solfegge_layer(x)
        
        return self.time_distributed(x)
    
    def apply_solfegge_constraints(self, sequence):
        """Apply solfege rules to generated sequence"""
        # Implement your solfege rules here based on your expertise
        # Example: Avoid large pitch jumps between consecutive notes
        for i in range(1, len(sequence)):
            if abs(sequence[i] - sequence[i-1]) > 7:  # Avoid jumps larger than a fifth
                sequence[i] = sequence[i-1] + random.choice([-3, -2, 2, 3, 4, 5])
        return sequence

# ===== الدالة الرئيسية =====
def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Configure TensorFlow
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.set_visible_devices([gpus[args.gpu]], "GPU")
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
            except RuntimeError as e:
                logging.error(f"GPU configuration error: {str(e)}")

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / "train.log",
        quiet=args.quiet,
    )
    tf.get_logger().setLevel(logging.INFO)

    # Set random seeds
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

    # Log command-line arguments
    logging.info("Running with command-line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"- {arg}: {value}")

    # === Data ===
    dataset_config = CONFIG[args.dataset]
    n_tracks = len(dataset_config["programs"])
    
    # Load training data
    logging.info("Loading training data...")
    train_data = load_json_data(args.input_dir, "train", args.max_beat, dataset_config)
    if not train_data["time"]:
        logging.error("No training data found! Exiting.")
        return
    
    # Load validation data
    logging.info("Loading validation data...")
    val_data = load_json_data(args.input_dir, "valid", args.max_beat, dataset_config)
    if not val_data["time"]:
        logging.error("No validation data found! Exiting.")
        return
    
    # Analyze datasets
    analyze_data(train_data, n_tracks, "Training")
    analyze_data(val_data, n_tracks, "Validation")
    
    # Create datasets
    train_dataset = create_dataset(train_data, n_tracks, args, training=True)
    val_dataset = create_dataset(val_data, n_tracks, args, training=False)
    
    # Calculate steps per epoch
    train_sequences = len(train_data["label"])
    val_sequences = len(val_data["label"])

    steps_per_epoch = max(train_sequences // args.batch_size, 10)
    validation_steps = max(val_sequences // args.batch_size, 1)

    logging.info(f"Auto-calculated steps_per_epoch: {steps_per_epoch}")
    logging.info(f"Auto-calculated validation_steps: {validation_steps}")

    # Prepare datasets for training
    train_dataset = (
        train_dataset
        .repeat()
        .shuffle(1000)
        .padded_batch(
            args.batch_size,
            padded_shapes=(
                {"time": [None], "pitch": [None], **({"duration": [None]} if args.use_duration else {})},
                [None]
            ),
            padding_values=(
                {"time": 0, "pitch": 0, **({"duration": 0} if args.use_duration else {})},
                -1
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .repeat()
        .padded_batch(
            args.batch_size,
            padded_shapes=(
                {"time": [None], "pitch": [None], **({"duration": [None]} if args.use_duration else {})},
                [None]
            ),
            padding_values=(
                {"time": 0, "pitch": 0, **({"duration": 0} if args.use_duration else {})},
                -1
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    # === Model ===
    logging.info("Building simplified LSTM model...")
    
    # Build the model
    model = LSTMArranger(
        use_duration=args.use_duration,
        n_tracks=n_tracks,
        d_model=args.d_model,
        n_layers=args.n_layers
    )
    
    # Build with sample input
    sample_input = {
        "time": tf.keras.Input(shape=(None,), dtype=tf.int32, name="time"),
        "pitch": tf.keras.Input(shape=(None,), dtype=tf.int32, name="pitch"),
    }
    if args.use_duration:
        sample_input["duration"] = tf.keras.Input(shape=(None,), dtype=tf.int32, name="duration")
    
    model(sample_input)
    
    # Log model summary
    model.summary(print_fn=logging.info)

    # === Compile the model ===
    logging.info("Compiling model...")
    
    def masked_loss(y_true, y_pred):
        """Custom loss function that ignores padded values."""
        # Create mask where 1 = real data, 0 = padded data
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
        
        # Clip labels to valid range
        y_true_adjusted = tf.clip_by_value(y_true, 0, n_tracks - 1)
        
        # Calculate loss per element
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
        loss = loss_fn(y_true_adjusted, y_pred)
        
        # Apply mask to ignore padded values
        loss = loss * mask
        
        # Calculate mean loss over non-padded elements
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1)

    def masked_acc(y_true, y_pred):
        """Custom accuracy function that ignores padded values."""
        # Convert logits to predicted labels
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        
        # Create mask where 1 = real data, 0 = padded data
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
        
        # Clip labels to valid range
        y_true_adjusted = tf.clip_by_value(y_true, 0, n_tracks - 1)
        
        # Calculate accuracy per element
        accuracies = tf.cast(tf.equal(y_true_adjusted, y_pred_labels), tf.float32)
        
        # Apply mask to ignore padded values
        accuracies = accuracies * mask
        
        # Calculate mean accuracy over non-padded elements
        return tf.reduce_sum(accuracies) / tf.maximum(tf.reduce_sum(mask), 1)

    # Optimizer with learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=args.steps_per_epoch * 10,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_acc]
    )

    # === Training ===
    logging.info("Training model...")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_dir / "best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=args.patience,
            monitor="val_loss",
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_history.csv"),
        tf.keras.callbacks.TensorBoard(
            log_dir=args.output_dir / "logs",
            histogram_freq=1
        )
    ]
    
    try:
    # ✅ تحميل checkpoint إن وُجد
      if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
          print(f"🔁 Loading weights from checkpoint: {args.resume_from_checkpoint}")
          model.load_weights(args.resume_from_checkpoint)

    # ✅ بدء التدريب
          history = model.fit(
              train_dataset,
              epochs=args.epoch,
              steps_per_epoch=args.steps_per_epoch,
              validation_data=val_dataset,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=1 if not args.quiet else 0,
          )

    # ✅ حفظ أوزان النموذج بعد التدريب للاستئناف لاحقًا
          model.save_weights(args.output_dir / "resume_checkpoint.keras")

except Exception as e:
    logging.error(f"Training failed: {str(e)}")
    return

    # Save final model
    model.save(args.output_dir / "final_model.keras")
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()