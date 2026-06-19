# 🎵 Automatic Instrumentation (Music Arrangement)

Graduation Project Submitted for the Bachelor's Degree in Artificial Intelligence
Faculty of Informatics Engineering – Tishreen University – Syria

## 📖 About the Project

This project aims to develop an intelligent system capable of automatically distributing musical notes across different instruments, starting from a monophonic musical piece (such as a solo piano performance).

The idea is inspired by the *keyboard zone splitting* feature found in modern electronic keyboards. However, this work extends the concept by making the assignment process dynamic, allowing the model to determine the most appropriate instrument for each note based on its musical context, including pitch, timing, and relationships with neighboring notes.

Three deep learning architectures were designed and evaluated:

* **LSTM** – captures temporal dependencies in musical sequences.
* **Transformer** – models long-range contextual relationships.
* **Hybrid Model (LSTM + Transformer)** – combines the strengths of both architectures.

The models were evaluated using the **Bach Chorales** and **Lakh MIDI Dataset (LMD)** datasets. Experimental results demonstrated that the hybrid architecture achieved the best balance between learning capacity and generalization performance.

---

## 🎼 System Pipeline

```text
MIDI Input
      ↓
Preprocessing
      ↓
LSTM / Transformer / Hybrid
      ↓
Instrument Prediction
      ↓
Arranged MIDI Output
```

---

## ✨ Key Features

* 🧠 Multiple Deep Learning Architectures (LSTM, Transformer, Hybrid)
* 🎼 Support for Different Musical Styles and Datasets
* ⏱️ Sequence-to-Sequence Processing of Musical Notes
* 🎹 Automatic Instrument Assignment for MIDI Arrangements
* 📊 Custom Evaluation Metrics with Masking Support
* 🚀 Easy Training and Inference Workflow

---

## 📂 Project Structure

```text
Auto-Music-Arranger/
├── README.md
├── LICENSE
├── .gitignore
├── lstm.py                # LSTM architecture
├── transformer.py         # Transformer architecture
├── hybrid.py             # Hybrid (LSTM + Transformer) architecture
├── preprocess_bach.py    # Bach dataset preprocessing
├── preprocess_lmd.py     # LMD dataset preprocessing
└── predict.py            # MIDI arrangement inference
```

---

## ⚙️ Requirements

Install the required libraries:

```bash
pip install tensorflow numpy muspy mido tqdm joblib pandas pathlib
```

Or install them using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Data Preparation

The datasets should be converted into JSON format before training.

Example structure:

```text
data/
└── splits/
    └── bach/
        ├── train/
        └── valid/
```

Each JSON file represents a musical piece containing note and instrument information used during training.

---

## 🚀 Training

Three different architectures are available:

### LSTM Model

```bash
python lstm.py
```

### Transformer Model

```bash
python transformer.py
```

### Hybrid Model

```bash
python hybrid.py
```

---

## 🎹 Inference

After training, generate an arranged MIDI file:

```bash
python predict.py
```

The system takes a MIDI file as input and produces a multi-track arranged MIDI output.

---

## 📊 Experimental Results

| Model                       | Training Accuracy | Validation Accuracy | Notes                                              |
| --------------------------- | ----------------- | ------------------- | -------------------------------------------------- |
| LSTM                        | ~95.1%            | ~89.4%              | Stable validation performance                      |
| Transformer                 | ~94.7%            | ~83.3%              | Strong training performance, weaker generalization |
| Hybrid (LSTM + Transformer) | ~92.6%            | ~92.0%              | Best balance between learning and generalization   |

### 📌 Conclusion

The Hybrid model achieved the best overall performance, combining the temporal modeling capability of LSTM networks with the long-range contextual understanding of Transformer architectures.

---

## 🔮 Future Directions

* Generate entirely new musical arrangements using generative models.
* Support real-time arrangement during live performances.
* Explore unsupervised and self-supervised learning techniques.
* Incorporate additional musical features such as rhythm, harmony, and musical modes.

---

## 👩‍🎓 Author

**Hanan Mohammad Hatim**
Artificial Intelligence Department
Faculty of Informatics Engineering – Tishreen University

**Supervisor:** Dr. Mohammad Mashi

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Special thanks to everyone who contributed to this work, and to the open-source community, especially TensorFlow, MusPy, NumPy, and MIDO.

---

⭐ If you find this project useful, consider giving it a star.
