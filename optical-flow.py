import streamlit as st
import tempfile

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.ticker import MultipleLocator

import sys

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .stImage {
            max-width: 100%;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

sys.path.append('/workspaces/BackgroundRemoval/raft-pytorch')

from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig

config = RAFTConfig(
    dropout=0,
    alternate_corr=False,
    small=False,
    mixed_precision=False
)

model = RAFT(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

weights_path = '/workspaces/BackgroundRemoval/raft-pytorch/raft-sintel.pth'

ckpt = torch.load(weights_path, map_location=device)
model.to(device)
model.load_state_dict(ckpt)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_video(uploaded_file):
    # Convert the uploaded file to bytes
    # video_bytes = uploaded_file.read()

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frames = []
    while True:
        has_frame, image = cap.read()

        if has_frame:
            image = image[:, :, ::-1]  # convert BGR -> RGB
            frames.append(image)
        else:
            break

    frames = np.stack(frames, axis=0)

    return frames

def adjust_pixel_values(image_array, low_adjustment, high_adjustment):
    # Ensure the image array is not empty
    if image_array is not None:
        # Convert the image array to float32 for better precision
        image_array = image_array.astype(np.float32)

        lower_pixels = image_array < 128
        higher_pixels = ~lower_pixels

        # Increase lower pixel values
        image_array[lower_pixels] = np.clip(image_array[lower_pixels] - low_adjustment, 0, 255)

        # Increase higher pixel values
        image_array[higher_pixels] = np.clip(image_array[higher_pixels] + high_adjustment, 0, 255)

        # Convert back to uint8
        image_array = image_array.astype(np.uint8)

        return image_array

    else:
        print("Error: Image array is empty.")

def viz(img1, img2, flo):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 6))
    ax1.set_title('input image1')
    ax1.imshow(img1.astype(int))
    ax2.set_title('input image2')
    ax2.imshow(img2.astype(int))
    ax3.set_title('estimated optical flow')

    flo = adjust_pixel_values(flo, 30, 30)
    ax3.imshow(flo)

    # grid
    num_lines = 5
    ax3.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
    ax3.xaxis.set_major_locator(MultipleLocator(len(flo[0]) // (num_lines)))
    ax3.yaxis.set_major_locator(MultipleLocator(len(flo) // (num_lines)))


    # printing text
    ax3.text(1.05, 0.5, sum(flo.flatten()), transform=ax3.transAxes, fontsize=12, verticalalignment='center')

    # plt.show()
    
    st.pyplot(fig)

def get_flow(n_vis = 90):
    
    start_at = 20

    for i in range(start_at,n_vis, 5):
        image1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().to(device)
        image2 = torch.from_numpy(frames[i+1]).permute(2, 0, 1).float().to(device)

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        print(image1.shape)

        with torch.no_grad():
            flow_low, flow_up = model(image1, image2, iters=5, test_mode=True)

        viz(image1, image2, flow_up)


st.title("Smart Cane Streamlit App")

uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mkv"])

if uploaded_file is not None:

    example_input = torch.randn(1, 3, 480, 272)
    print(type(model))
    # Export the model to ONNX
    torch.onnx.export(model, example_input, "model.onnx", export_params=True, opset_version=11)

    # Process and display the video
    frames = process_video(uploaded_file)

    st.write("Count of frames:", len(frames))

    get_flow()
