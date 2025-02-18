# Text-ai
a Facial recognition program that listens to the speech of the user and displays it as a comic text font on the forehead

## Features

*   **Real-time Face Detection:** Utilizes MediaPipe for accurate face detection in real-time video streams.
*   **Speech Recognition:** Employs OpenAI's Whisper model to transcribe spoken words into text.
*   **Dynamic Speech Bubble Generation:** Automatically generates and positions speech bubbles above detected faces, displaying the recognized text.
*   **Multithreading:** Implements multithreading for efficient video capture and audio processing, ensuring smooth and responsive performance.
*   **Frame Rate Control:** Manages the frame rate to maintain a consistent and visually appealing output.

## Technical Skills Demonstrated

*   **Computer Vision:**
    *   Video capture and frame manipulation using OpenCV (cv2).
    *   Face detection algorithms from MediaPipe.
    *   Image drawing and text rendering using OpenCV functions.
*   **Audio Processing:**
    *   Real-time audio recording using `sounddevice`.
    *   Audio normalization and format conversion with `pydub`.
    *   Integration with speech recognition models.
*   **Natural Language Processing:**
    *   Integration of pre-trained machine learning models (OpenAI's Whisper).
    *   Text processing and manipulation for displaying recognized speech.
*   **Multithreading:**
    *   Parallel processing of video and audio streams using Python's `threading` module.
    *   Thread synchronization and data sharing using locks and queues.
    *   Efficient resource utilization for real-time performance.
*   **User Interface:**
    *   Simple graphical interface created using OpenCV's `imshow` function.
    *   Real-time GUI updates to display the video feed and speech bubbles.
    *   Event handling (keyboard input) for exiting the application.
*   **Data Structures and Algorithms:**
    *   Queues (`queue.Queue`) for managing recognized text and synchronizing between threads.
    *   Lists for storing and updating displayed words, creating a dynamic text display.
    *   Custom algorithms for speech bubble placement and text wrapping.
*   **Library Integration:**
    *   Demonstrates effective integration and usage of multiple Python libraries, including `opencv-python`, `mediapipe`, `whisper`, `pydub`, and `sounddevice`.

## Getting Started

### Prerequisites

*   Python 3.6 or higher
*   pip package installer

### Installation

1.  Clone the repository:

    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required libraries:

    ```
    pip install opencv-python mediapipe whisper pydub sounddevice numpy
    ```

### Usage

1.  Run the script:

    ```
    python Text-ai.py
    ```

2.  The application will open your webcam and start detecting faces and recognizing speech.

3.  Speak clearly, and your words will appear in a speech bubble above the detected face.

4.  Press 'q' to quit the application.

## Code Structure

*   `VideoCaptureThread`: A class that handles video capture in a separate thread.
*   `WhisperThread`: A class that handles audio recording, processing, and speech recognition in a separate thread.
*   `draw_speech_bubble`: A function that generates and draws the speech bubble on the video frame.
*   The main loop captures video frames, detects faces, recognizes speech, and displays the results in real-time.

## Notes

*   The "base" Whisper model is used for speech recognition. You can experiment with other models ("small", "medium", or "large") for potentially better accuracy, but they may require more computational resources.
*   The application processes every other frame to reduce the workload and improve performance. This can be adjusted based on your hardware capabilities.
*   This project serves as a demonstration of various Python programming skills and library integrations. It can be further improved and optimized for specific use cases or performance requirements.

## Further Development

Possible enhancements and future directions for this project include:

*   Improve speech recognition accuracy by fine-tuning the Whisper model or using other advanced techniques.
*   Implement more sophisticated speech bubble designs and animations.
*   Add support for multiple faces and speech bubbles.
*   Create a more user-friendly interface with customizable options.
*   Incorporate additional comic book-style effects and filters.
