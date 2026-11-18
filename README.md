# Real-Time Hand Gesture Control
 
Webcam-based interface that recognizes hand gestures and uses them to create, move, and delete on-screen boxes. Final project for CSE 576 (Computer Vision), University of Washington.
 
## Approach
 
- **MediaPipe** detects 21 hand landmarks per hand from the webcam feed.
- A **multilayer perceptron (MLP)** classifies the landmarks into 8 gestures (like, dislike, fist, hand heart, ok, peace, point, stop).
- Trained on a subsample of the [HaGRID](https://doi.org/10.48550/arXiv.2206.08219) dataset (~36,800 landmark samples).
- Gesture combinations from both hands trigger commands, e.g., fist + like creates a box.
## Results
 
- **93.5% accuracy** and 0.94 weighted F1 on a held-out test set (PyTorch MLP with batch normalization and dropout).
- Model is under 10 MB and runs in real time on a standard laptop.
- Most confusion was between gestures with similar finger shapes (point, peace, dislike).

## Team

- **Bauyrzhan Zhandulla:** MediaPipe hand landmark pipeline from the webcam
- **Zelamir Reynolds-Henkins:** gesture-controlled commands
- **Oviya Muralidharan and Katie Sie:** gesture classifier model architecture and training
