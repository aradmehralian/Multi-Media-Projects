import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import json

st.title("Webcam Live Preview and Image Classification")

# Initialize session state if not already done
if 'started' not in st.session_state:
    st.session_state.started = False
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'predicted_label' not in st.session_state:
    st.session_state.predicted_label = None
if 'capture_flag' not in st.session_state:
    st.session_state.capture_flag = False
if 'upload_flag' not in st.session_state:
    st.session_state.upload_flag = False
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'clear_flag' not in st.session_state:
    st.session_state.clear_flag = False

# Reset session state on app rerun
if not st.session_state.started and not st.session_state.capture_flag and not st.session_state.upload_flag:
    st.session_state.predicted_label = None
    st.session_state.frame = None
    st.session_state.file_name = ""

# Load the pretrained ResNet-50 model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

# ImageNet class labels
@st.cache_resource
def load_labels():
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(LABELS_URL)
    labels = json.loads(response.text)
    return labels

labels = load_labels()

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0)
    return image

# Function to get webcam feed
def get_frame(cap):
    ret, frame = cap.read()
    return ret, frame

# Function to start webcam
def start_webcam():
    st.session_state.started = True
    st.session_state.capture_flag = False
    st.session_state.upload_flag = False

# Function to stop webcam
def stop_webcam():
    st.session_state.started = False

# Function to capture and process image
def capture_image():
    if st.session_state.frame is not None:
        st.session_state.capture_flag = True
        st.session_state.started = False

# Function to save the captured image and process it
def save_and_process_image():
    if st.session_state.frame is not None and st.session_state.file_name:
        file_name = st.session_state.file_name
        if not os.path.splitext(file_name)[1]:  # if no extension is provided
            file_name += ".jpg"
        image_path = os.path.join(os.getcwd(), file_name)
        cv2.imwrite(image_path, cv2.cvtColor(st.session_state.frame, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved at {image_path}")

        # Preprocess the image
        image = preprocess_image(st.session_state.frame)

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
        _, predicted = outputs.max(1)
        st.session_state.predicted_label = labels[predicted.item()]

# Function to upload and process image
def upload_image(image_file):
    st.session_state.upload_flag = True
    img = Image.open(image_file)
    st.session_state.frame = np.array(img)

    # Display processing message
    processing_placeholder.text("Processing image...")

    # Preprocess the image
    image = preprocess_image(st.session_state.frame)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    st.session_state.predicted_label = labels[predicted.item()]

    # Remove processing message
    processing_placeholder.empty()

# Function to clear the output
def clear_output():
    st.session_state.frame = None
    st.session_state.predicted_label = None
    st.session_state.capture_flag = False
    st.session_state.upload_flag = False
    st.session_state.file_name = ""
    st.session_state.clear_flag = True

# Placeholders for UI elements
processing_placeholder = st.empty()
frame_placeholder = st.empty()

# Buttons to control webcam and capture image
start_button = st.button("Start Webcam", on_click=start_webcam, key="start")
stop_button = st.button("Stop Webcam", on_click=stop_webcam, key="stop")
capture_button = st.button("Capture Image", on_click=capture_image, key="capture")
clear_button = st.button("Clear Output", on_click=clear_output, key="clear")

# File uploader for pre-existing images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")

# Handle file name input and save button after capturing the image
if st.session_state.capture_flag:
    st.session_state.file_name = st.text_input("Enter the name to save the file (default extension is .jpg):")
    if st.button("Save Image"):
        save_and_process_image()
        st.session_state.capture_flag = False  # Reset capture flag after saving and processing

# Display the captured or uploaded image and the predicted label if available
captured_image_placeholder = st.empty()
predicted_label_placeholder = st.empty()

if not st.session_state.clear_flag:
    if st.session_state.frame is not None:
        captured_image_placeholder.image(st.session_state.frame, caption="Captured Image", channels="RGB")
    if st.session_state.predicted_label is not None:
        predicted_label_placeholder.write(f"Predicted Label: {st.session_state.predicted_label}")
else:
    st.session_state.clear_flag = False

if uploaded_file is not None:
    upload_image(uploaded_file)

# Initialize the webcam
if st.session_state.started:
    cap = cv2.VideoCapture(0)
    while st.session_state.started:
        ret, frame = get_frame(cap)
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.frame = frame_rgb
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Adding a small delay to avoid high CPU usage
        cv2.waitKey(1)
    
    cap.release()
