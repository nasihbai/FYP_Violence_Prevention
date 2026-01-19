# LSTM Actions Recognition Project Analysis

## Project Overview

This is an **LSTM-based Action Recognition System** that uses MediaPipe for pose/hand landmark detection and TensorFlow/Keras for temporal action classification. The system processes video frames in real-time to detect and classify human actions.

## Core Technologies

- **MediaPipe**: For pose and hand landmark extraction
- **TensorFlow/Keras**: LSTM neural network for temporal sequence classification
- **OpenCV**: Video capture and processing
- **NumPy & Pandas**: Data manipulation

## Project Components

### 1. **Data Generation Scripts**
- `pose_data_generation.py` - Captures pose landmarks for body action recognition
- `hands_data_generation.py` - Captures hand landmarks for hand gesture recognition

### 2. **Model Training**
- `train_model.py` - Trains LSTM model on collected landmark data

### 3. **Real-time Detection Scripts**
- `pose_lstm_realtime.py` - Real-time pose-based action detection (violent/neutral)
- `hands_lstm_realtime.py` - Real-time hand gesture detection (grasping actions)
- `hands_lstm_realtime_custom.py` - Customized hand detection variant

## Action Categories Detected

### Hand Gestures (Grasp Recognition)
The project tracks different types of hand grasping actions:

1. **Carrying** - Hand holding/carrying an object with the whole hand
2. **Cupping** - Hand forming a cup shape to hold objects
3. **Grasping** - Active grasping motion with fingers closing around object
4. **Gripping** - Firm grip with fingers wrapped around object
5. **Holding** - Static holding position maintaining object
6. **Resting** - Hand in rest position without holding anything

### Body Actions (Pose Recognition)
1. **Violent** - Detecting aggressive or violent body movements
2. **Neutral** - Normal, non-violent body posture and movement

## How It Works

### Architecture Flow

```
Video Input → MediaPipe Landmarks → Temporal Buffer (20 frames) → LSTM Model → Action Classification
```

### Detailed Process

1. **Landmark Extraction**
   - MediaPipe extracts 33 pose landmarks (x, y, z, visibility) or 21 hand landmarks (x, y, z)
   - Each landmark provides spatial coordinates in normalized space

2. **Temporal Buffering**
   - System collects 20 consecutive frames of landmarks
   - Creates a temporal sequence: `[20 timesteps × number of features]`
   - This captures motion patterns over time

3. **LSTM Processing**
   - 4-layer LSTM network with dropout (0.2) for regularization
   - Architecture: 50 units per LSTM layer
   - Final dense layer with softmax activation for classification
   - Processes temporal sequences to recognize action patterns

4. **Real-time Prediction**
   - Threading used to run predictions without blocking video stream
   - Warm-up period (60 frames) to stabilize detection
   - Continuous sliding window of 20 frames for ongoing classification

### Model Structure (from train_model.py)

```
LSTM Layer 1: 50 units, return_sequences=True
Dropout: 0.2
LSTM Layer 2: 50 units, return_sequences=True
Dropout: 0.2
LSTM Layer 3: 50 units, return_sequences=True
Dropout: 0.2
LSTM Layer 4: 50 units
Dropout: 0.2
Dense Layer: 4 units, softmax activation
```

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 100
- **Batch Size**: 32
- **Train/Test Split**: 80/20

## File Structure

```
├── pose_lstm_realtime.py          # Pose detection demo (violent detection)
├── hands_lstm_realtime.py         # Hand gesture detection demo
├── hands_lstm_realtime_custom.py  # Custom hand detection variant
├── pose_data_generation.py        # Generate pose training data
├── hands_data_generation.py       # Generate hand training data
├── train_model.py                 # Train LSTM model
├── lstm-model.h5                  # Trained pose model (violent detection)
├── lstm-hand-gripping.h5          # Trained hand grasping model
├── lstm-hand-grasping.h5          # Alternative hand model
├── model.h5                       # General model
└── Dataset files (.txt):
    ├── carrying.txt               # Carrying action dataset
    ├── cupping.txt                # Cupping action dataset
    ├── grasping.txt               # Grasping action dataset
    ├── gripping.txt               # Gripping action dataset
    ├── holding.txt                # Holding action dataset
    ├── resting.txt                # Resting state dataset
    ├── neutral.txt                # Neutral pose dataset
    └── violent.txt                # Violent action dataset
```

## Key Features

### 1. **Temporal Sequence Learning**
- LSTM networks capture temporal dependencies across 20 frames
- Recognizes action patterns, not just static poses
- Distinguishes between similar poses based on motion

### 2. **Real-time Processing**
- Threading prevents prediction lag from blocking video stream
- 60-frame warm-up ensures stable detection
- Continuous sliding window for seamless classification

### 3. **Visual Feedback**
- Bounding boxes around detected subjects
- Color-coded labels (green for neutral, red for action)
- Landmark visualization for debugging
- Status text overlay on video feed

### 4. **Customizable Architecture**
- Modular design allows easy addition of new action classes
- Adjustable confidence thresholds
- Configurable temporal window size (default: 20 frames)

## Use Cases

1. **Violence Detection** - Security monitoring, surveillance systems
2. **Hand Gesture Recognition** - Object manipulation tracking, rehabilitation, HCI
3. **Activity Monitoring** - Fitness tracking, ergonomics analysis
4. **Behavioral Analysis** - Research, psychology studies

## Technical Specifications

### Input Requirements
- Video feed: RGB frames
- Frame rate: Real-time (typically 30 FPS)
- Resolution: Flexible (automatically normalized by MediaPipe)
- Camera index: Configurable (default 0 for webcam, 6 for external device)

### Model Input Shape
- **Pose**: `(20, 132)` - 20 timesteps × (33 landmarks × 4 features)
- **Hands**: `(20, 63)` - 20 timesteps × (21 landmarks × 3 features)

### Output
- Softmax probabilities for each action class
- Threshold: 0.5 (50% confidence) for classification
- Real-time label display on video

## Workflow Summary

### Training Workflow
1. Run data generation script → Perform actions → Generate .txt dataset
2. Configure train_model.py with datasets and labels
3. Train model → Save as .h5 file
4. Test with real-time detection script

### Inference Workflow
1. Load trained model from .h5 file
2. Initialize video capture and MediaPipe
3. Process frames → Extract landmarks
4. Buffer 20 frames → Run LSTM prediction
5. Display results with bounding boxes and labels

## Performance Considerations

- **Warm-up Period**: 60 frames ensures stable landmark detection
- **Threading**: Prevents prediction lag from affecting frame rate
- **Model Size**: 4 LSTM layers balance accuracy and speed
- **Dropout**: 0.2 rate prevents overfitting on small datasets

## Customization Points

1. **Number of classes**: Modify dataset loading and output layer
2. **Temporal window**: Adjust `no_of_timesteps` (default: 20)
3. **Confidence threshold**: Change from 0.5 to tune sensitivity
4. **LSTM architecture**: Add/remove layers, adjust units
5. **Visual styling**: Customize colors, fonts, bounding boxes

---

## Conclusion

This project demonstrates effective use of LSTM networks for temporal action recognition, combining MediaPipe's robust landmark detection with deep learning for real-time classification. The modular design allows easy adaptation to various action recognition tasks.
