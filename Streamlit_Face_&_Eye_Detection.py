import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# WebRTC configuration (optional, for better connection handling)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Face & Eye Detection", page_icon="👁️", layout="wide")

def load_cascade_classifier(file_path):
    """Load a Haar Cascade classifier from the given file path."""
    if not os.path.exists(file_path):
        st.error(f"Error: Cascade classifier file {file_path} not found!")
        return None
    classifier = cv2.CascadeClassifier(file_path)
    if classifier.empty():
        st.error(f"Error: Failed to load cascade classifier from {file_path}!")
        return None
    return classifier

def detect_faces_and_eyes(image):
    """Detect faces and eyes in the image and draw rectangles around them."""
    # Convert image to OpenCV format (BGR)
    if isinstance(image, np.ndarray):
        image_cv = image
    else:
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Detect faces with fixed parameters
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        st.warning("No faces detected in this image!")
        return image_cv

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (127, 0, 255), 2)
        
        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image_cv[y:y + h, x:x + w]
        
        # Detect eyes
        eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
    
    return image_cv

class VideoProcessor(VideoProcessorBase):
    """Class to process video frames for face and eye detection."""
    def __init__(self):
        self.face_classifier = face_classifier
        self.eye_classifier = eye_classifier

    def recv(self, frame):
        # Convert frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame for face and eye detection
        processed_img = detect_faces_and_eyes(img)
        
        # Return processed frame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    st.title("🙂Face and 👁️Eye Detection App")
    st.write("Choose an option in the sidebar to either upload images or use live video for face and eye detection.")

    # Load Haar Cascade classifiers with specified paths
    global face_classifier, eye_classifier
    face_cascade_path = r"C:\Users\umesh\FS_Data_Science\AI\OpenCV\HaarCascade\haarcascade_frontalface_default.xml"
    eye_cascade_path = r"C:\Users\umesh\FS_Data_Science\AI\OpenCV\HaarCascade\haarcascade_eye.xml"
    
    face_classifier = load_cascade_classifier(face_cascade_path)
    eye_classifier = load_cascade_classifier(eye_cascade_path)
    
    if face_classifier is None or eye_classifier is None:
        return

    # Sidebar for mode selection and file uploader
    st.sidebar.title("Detection Options")
    mode = st.sidebar.radio("Select Mode", ["Image Upload", "Live Video"])

    if mode == "Image Upload":
        st.write("Upload one or multiple images to detect faces and eyes using Haar Cascade classifiers.")
        uploaded_files = st.sidebar.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            for idx, uploaded_file in enumerate(uploaded_files):
                # Display image name using markdown
                st.markdown(f"Image {idx + 1}: {uploaded_file.name}")
                
                # Load the uploaded image
                image = Image.open(uploaded_file)
                
                # Create two columns for side-by-side display
                col1, col2 = st.columns([0.5,2])
                
                # Display uploaded image in the first column with a smaller width
                with col1:
                    st.image(image, caption="Uploaded Image", width=200)
                
                # Detect faces and eyes
                processed_image = detect_faces_and_eyes(image)
                
                # Convert back to RGB for Streamlit display
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                # Display processed image in the second column with a larger width
                with col2:
                    st.image(processed_image_rgb, caption="Processed Image with Face and Eye Detection", width=800)
                
                # Add a horizontal line to separate multiple images
                if idx < len(uploaded_files) - 1:
                    st.markdown("---")
    
    elif mode == "Live Video":
        st.write("Use your webcam for real-time face and eye detection. Ensure your browser has webcam access enabled.")
        try:
            webrtc_streamer(
                key="face-eye-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")

if __name__ == "__main__":
    main()