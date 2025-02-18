import cv2
import mediapipe as mp
import threading
import time
import queue
import whisper
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import sounddevice as sd  # For recording audio

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Whisper setup
model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy

# Video capture thread
class VideoCaptureThread:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# Audio preprocessing and Whisper thread
class WhisperThread:
    def __init__(self):
        self.running = True
        self.text_queue = queue.Queue()

    def start(self):
        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        sample_rate = 16000  # Whisper works best with 16kHz audio
        channels = 1  # Mono audio

        while self.running:
            print("Recording...")
            duration = 5  # Record for 5 seconds
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
            sd.wait()  # Wait for recording to finish

            # Convert numpy array to AudioSegment
            audio = np.int16(audio * 32767)  # Convert to 16-bit PCM
            audio = AudioSegment(
                audio.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes
                channels=channels
            )

            # Preprocess audio
            audio = normalize(audio)  # Normalize volume
            audio = audio.set_frame_rate(sample_rate)  # Ensure correct sample rate

            # Export to WAV file
            audio.export("temp_audio.wav", format="wav")

            # Load audio using pydub and convert to numpy array
            audio_segment = AudioSegment.from_file("temp_audio.wav")
            samples = np.array(audio_segment.get_array_of_samples())
            samples = samples.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Transcribe using Whisper
            result = model.transcribe(samples)
            text = result["text"]
            self.text_queue.put(text)
            print("You said:", text)

    def stop(self):
        self.running = False

# Initialize video capture and Whisper threads
capture_thread = VideoCaptureThread()
capture_thread.start()

whisper_thread = WhisperThread()
whisper_thread.start()

# Frame rate control
target_fps = 15
frame_delay = 1 / target_fps

# Speech bubble settings
bubble_color = (255, 255, 255)  # White
text_color = (0, 0, 0)  # Black

# List to store the last 7 words
word_list = []

# Speech bubble function
def draw_speech_bubble(frame, text, face_center):
    bubble_padding = 10
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_width, text_height = text_size
    bubble_width = text_width + 2 * bubble_padding
    bubble_height = text_height + 2 * bubble_padding

    bubble_x = face_center[0] - bubble_width // 2
    bubble_y = face_center[1] - bubble_height - 50

    cv2.rectangle(
        frame,
        (bubble_x, bubble_y),
        (bubble_x + bubble_width, bubble_y + bubble_height),
        bubble_color,
        -1,
    )
    cv2.rectangle(
        frame,
        (bubble_x, bubble_y),
        (bubble_x + bubble_width, bubble_y + bubble_height),
        (0, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        text,
        (bubble_x + bubble_padding, bubble_y + bubble_padding + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
    )

    return frame

# Main loop
process_frame = False
while True:
    start_time = time.time()

    # Read frame from the video capture thread
    frame = capture_thread.read()
    if frame is None:
        continue

    # Process every other frame to reduce workload
    process_frame = not process_frame
    if process_frame:
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = face_detection.process(rgb_frame)

        # Get face center (if detected)
        face_center = None
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame.shape[1])
                y = int(bbox.ymin * frame.shape[0])
                w = int(bbox.width * frame.shape[1])
                h = int(bbox.height * frame.shape[0])
                face_center = (x + w // 2, y + h // 2)

        # Check for recognized text
        if not whisper_thread.text_queue.empty():
            new_text = whisper_thread.text_queue.get()
            words = new_text.split()  # Split the recognized text into words
            for word in words:
                word_list.append(word)  # Add each word to the list
                if len(word_list) > 7:  # Keep only the last 7 words
                    word_list.pop(0)  # Remove the oldest word
            current_text = " ".join(word_list)  # Combine words into a single string
            print("You said:", current_text)

        # Draw speech bubble if face is detected
        if face_center and word_list:
            frame = draw_speech_bubble(frame, current_text, face_center)

    # Display the frame
    cv2.imshow('Comic Book AI', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Control frame rate
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_delay:
        time.sleep(frame_delay - elapsed_time)

# Cleanup
capture_thread.stop()
whisper_thread.stop()
cv2.destroyAllWindows()