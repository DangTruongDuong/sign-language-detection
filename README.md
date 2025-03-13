# Hand Gesture Recognition Project

Real-time hand gesture recognition system using computer vision and machine learning.

This project implements a hand gesture recognition system that captures hand movements via a webcam, processes them with computer vision techniques, and translates them into text (e.g., alphabet letters) using a machine learning model. Built with Python, OpenCV, MediaPipe, and scikit-learn, it leverages a Random Forest Classifier for gesture classification.

## 🚀 Features
- **Data Collection**: Capture hand gesture images using a webcam.
- **Feature Extraction**: Extract hand landmarks with MediaPipe.
- **Model Training**: Train a Random Forest Classifier for gesture recognition.
- **Real-Time Detection**: Recognize gestures in real-time and output text.
- **Customizable**: Supports multiple gestures (e.g., A-Y, extensible to more).

## 📋 Requirements
- **Python**: 3.8 or higher
- **Hardware**: Webcam (built-in or external)
- **Dependencies**:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `cvzone`
  - `pickle`

## 🛠 Installation

### Clone the Repository:
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is not provided, install manually:
```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib cvzone
```

### Verify Webcam:
Ensure your webcam is connected and operational.

## 📖 Usage

### Step 1: Collect Data
Capture hand gesture images for training:
```bash
python data_collection.py
```
- Creates a `data/` folder with subdirectories (`0`, `1`, etc.) for each class.
- Captures 200 images per class (adjustable via `dataset_size`).
- **Instructions**: Press `Q` to start collecting data for each class.

### Step 2: Process Data
Extract hand landmarks and save to a pickle file:
```bash
python data_processing.py
```
- Processes images in the `data/` folder.
- Uses MediaPipe to detect and normalize hand landmarks.
- Outputs `data.pickle` with processed data.

### Step 3: Train the Model
Train the Random Forest Classifier:
```bash
python model_training.py
```
- Loads data from `data.pickle`.
- Trains the model and evaluates accuracy.
- Saves the trained model to `model.p`.

### Step 4: Run Real-Time Recognition
Perform real-time gesture recognition:
```bash
python real_time_recognition.py
```
- Opens the webcam for live detection.
- Displays recognized characters and builds a text string.
- **Exit**: Press `Q` to close the application.

## 📂 File Structure
```
hand-gesture-recognition/
├── data/                   # Stores captured gesture images
├── data_collection.py      # Collects gesture images
├── data_processing.py      # Processes images into features
├── model_training.py       # Trains the machine learning model
├── real_time_recognition.py # Runs real-time gesture recognition
├── data.pickle             # Processed data file
├── model.p                 # Trained model file
├── README.md               # Project documentation
└── requirements.txt        # (Optional) Dependency list
```

## 🔍 How It Works
1. **Data Collection**: Captures webcam images of hand gestures and organizes them by class.
2. **Data Processing**: Detects hand landmarks using MediaPipe and normalizes coordinates.
3. **Model Training**: Trains a Random Forest Classifier on the processed data.
4. **Real-Time Recognition**: Detects gestures live, predicts labels, and converts them to text.

## ⚠️ Limitations
- Default setup supports only 2 classes (modify `number_of_classes` to extend).
- Accuracy depends on lighting, hand positioning, and training data quality.
- The `labels_dict` in `real_time_recognition.py` contains duplicate mappings (e.g., `0: 'A', 1: 'A'`); update it for a full alphabet system.

## 🤝 Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-idea`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-idea`).
5. Open a pull request.

**Suggestions for enhancements**:
- Improve model accuracy with more data or a different algorithm.
- Add support for additional gestures.
- Optimize real-time performance.

## 📜 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.
