markdown
# 🎵 Automatic Instrumentation (Music Arrangement)

> Graduation Project Submitted for the Bachelor's Degree in Artificial Intelligence  
> Faculty of Informatics Engineering – Tishreen University – Syria

---

## 📖 About the Project

This project aims to develop an intelligent system capable of **automatically distributing musical notes across different instruments**, starting from a monophonic musical piece (such as a solo piano performance).

The idea is inspired by the "keyboard zone splitting" feature found in modern electronic keyboards (organs). However, we extend this concept to make it **dynamic**, where the model decides the appropriate instrument for each note based on its musical context (time, pitch, and relationships with neighboring notes).

Three main models were proposed and trained:
- **LSTM** (to capture temporal dependencies).
- **Transformer** (to understand long-range context).
- **Hybrid Model** (combining both to achieve the best performance).

The models were evaluated on two datasets: **Bach Chorales** and **LMD (pop music)**. The results showed that the hybrid model outperformed the others in terms of generalization accuracy.

---

## ✨ Key Features

- 🧠 **Multiple Deep Models**: Support for LSTM, Transformer, and Hybrid architectures.
- 🎼 **Diverse Data Support**: Works on Bach chorales (4 voices) and LMD (Piano, Guitar, Bass, Strings, Brass).
- ⏱️ **Sequential Processing**: Handles notes as time series (Sequence-to-Sequence).
- 🛠️ **Flexible Interface**: Easy training and inference on real MIDI files.
- 📊 **Custom Metrics**: Uses `Masked Loss` and `Masked Accuracy` to ignore padded values.

---

## 📂 Project Structure (Core Files)

```text
Auto-Music-Arranger/
├── Trainlstm.py                # Train the LSTM model
├── Lstm+Transformer.py         # Train the Hybrid model
├── train.py                    # Train the Transformer model
├── predict.py                  # Inference on MIDI files (LSTM/Hybrid version)
├── transformerpredict.py       # Inference on MIDI files (Transformer version)
├── requirements.txt            # Required Python libraries
├── .gitignore                  # Files excluded from version control
├── LICENSE                     # MIT License
└── README.md                   # This file
⚙️ Requirements
Install the required libraries using pip:

bash
pip install tensorflow numpy muspy mido tqdm joblib pandas pathlib
Alternatively, you can use the requirements.txt file:

bash
pip install -r requirements.txt
🚀 How to Use
1️⃣ Data Preparation
Your data must be in JSON format, split into train and valid folders. Each JSON file should contain the musical piece with its tracks. Example data structure:

text
data/
└── splits/
    └── bach/
        ├── train/
        │   ├── piece_1.json
        │   └── piece_2.json
        └── valid/
            ├── piece_3.json
            └── piece_4.json
2️⃣ Training the Model
A. Train LSTM Model:

bash
python Trainlstm.py -i data/splits/bach -o results/bach_lstm -d bach -e 50 -bs 16
B. Train Hybrid Model (LSTM + Transformer):

bash
python Lstm+Transformer.py -i data/splits/lmd -o results/lmd_hybrid -d lmd -e 50 -bs 16
C. Train Transformer Model:

bash
python train.py -i data/splits/bach -o results/bach_transformer -d bach -e 50 -bs 16
Important Arguments:

-i, --input_dir: Path to the data folder.

-o, --output_dir: Path to save the model and logs.

-d, --dataset: Dataset type (bach or lmd).

-e, --epoch: Number of training epochs.

-bs, --batch_size: Batch size.

-g, --gpu: GPU device number (optional).

3️⃣ Inference on a MIDI File
After training, use the model to arrange a new piece:

bash
python predict.py -m results/bach_lstm/best_model.keras -i input.mid -o output_arranged.mid -d bach
Important Arguments:

-m, --model_path: Path to the trained model.

-i, --input_midi: Input MIDI file.

-o, --output_midi: Output MIDI file (will become multi-track).

-d, --dataset: Dataset type matching the model.

📊 Experimental Results
The models were trained on two datasets, and the results were as follows:

Model	Training Accuracy	Validation Accuracy	Notes
LSTM	~95.1%	~89.4%	More stable on validation
Transformer	~94.7%	~83.3%	Excellent training performance, weaker generalization
Hybrid (LSTM + Transformer)	~92.6%	~92.0%	Best balance between learning and generalization
📌 Conclusion: The hybrid model is the most suitable for this task, as it combines the LSTM's ability to capture temporal patterns with the Transformer's capability to understand long-range relationships between notes.

🔮 Future Directions
Using generative modeling to create entirely new arrangements.

Improving the model to work in real-time during live performances.

Applying unsupervised learning techniques to reduce the need for labeled data.

Incorporating additional features such as rhythm and musical modes to improve accuracy.

📄 License
This project is licensed under the MIT License – you are free to use, modify, and distribute it with proper attribution.

👩‍🎓 Author
Hanan Mohammad Hatim
Student, Department of Artificial Intelligence – Faculty of Informatics Engineering – Tishreen University
(Supervised by: Dr. Mohammad Mashi)

🙏 Acknowledgments
Special thanks to everyone who contributed to this work, and to the open-source libraries that made building these models easier, especially TensorFlow and MusPy.

⭐ Don't forget to give the project a star if you find it useful!
