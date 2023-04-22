import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

st.set_page_config(page_title="PicPerfector: Ultimate Photo Transformation",
                   page_icon=":camera_flash:",
                   layout="wide")
st.markdown("<h1 style='text-align: center; color: #b89b7b'>PicPerfector: Ultimate Photo Transformation</h1>", 
            unsafe_allow_html=True)

def improve_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def enhance_symmetry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    flipped = cv2.flip(image, 1)
    result = cv2.addWeighted(image, 0.5, flipped, 0.5, 0)
    return result

def adjust_background_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return result

def remove_stray_hairs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, binary = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.addWeighted(image, 1, cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR), -1, 0)
    return result

# Load StyleGAN2 from TensorFlow Hub
#stylegan2 = hub.load("https://tfhub.dev/google/stylegan2-ffhq-config-f/1")
progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

def generate_images(image, num_images=10, apply_lighting=False, apply_symmetry=False, apply_bg_color=False, apply_hair_removal=False):
    
    if selected_enhancements is None:
        selected_enhancements = [] # Set default value for selected_enhancements parameter
    
    for i in range(num_images):
        # Generate a random seed for the GAN
        seed = tf.random.normal([1, 512])

        # Generate an image using the GAN and the seed
        gan_output = progan(seed)

        # Convert the generated image back to the [0, 255] range
        generated_image = (gan_output + 1) / 2 * 255

        # Apply selected enhancements to the generated image
        generated_image = tf.squeeze(generated_image, axis=0).numpy().astype(np.uint8)
        if apply_lighting:
            generated_image = improve_lighting(generated_image)
        if apply_symmetry:
            generated_image = enhance_symmetry(generated_image)
        if apply_bg_color:
            generated_image = adjust_background_color(generated_image)
        if apply_hair_removal:
            generated_image = remove_stray_hairs(generated_image)

        generated_images.append(generated_image)

    return generated_images

def generate_new_images_based_on_feedback(selected_images):
    # Calculate the average image from the selected images
    average_image = np.mean(selected_images, axis=0).astype(np.uint8)

    # Generate new images using the average image as a base
    new_images = generate_images(average_image, num_images=len(selected_images))

    return new_images

# App Interface

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"]) # Set maximum upload size to 10 MB

if uploaded_file is not None:
    # Set maximum upload size to 10 MB
    uploaded_file.seek(0)
    max_size = 10 * 1024 * 1024
    if len(uploaded_file.getvalue()) > max_size:
        st.error(f"Please upload an image smaller than {max_size/1024/1024} MB.")
    else:
        input_image = Image.open(uploaded_file).convert("RGB")
        
        st.image(input_image, caption="Original Image")
        
        st.sidebar.info("Please select the enhancements to apply to the original image")
        
        #st.sidebar.help("Hover over each enhancement to see a brief description")

        enhance_lighting = st.sidebar.checkbox("Improve Lighting", help="Enhance brightness and contrast of the image")
        #enhance_symmetry = st.sidebar.checkbox("Enhance Facial Symmetry", help="Improve facial symmetry using reflection")
        if "enhance_symmetry" in selected_enhancements:
            generated_image = enhance_symmetry(generated_image)
        adjust_bg_color = st.sidebar.checkbox("Adjust Background Color", help="Change the background color of the image")
        remove_hairs = st.sidebar.checkbox("Remove Stray Hairs", help="Remove unwanted hairs from the image")

        selected_enhancements = []
        if enhance_lighting:
            selected_enhancements.append("improve_lighting")
        if apply_symmetry:
            selected_enhancements.append("enhance_symmetry")
        if adjust_bg_color:
            selected_enhancements.append("adjust_background_color")
        if remove_hairs:
            selected_enhancements.append("remove_stray_hairs")
            
        enhanced_images = []
        if st.button("Generate Enhanced Images"):
            #enhanced_images = generate_images(np.array(input_image), num_images=10, selected_enhancements=selected_enhancements)
            enhanced_images = generate_images(np.array(input_image), num_images=10, apply_lighting=enhance_lighting, apply_symmetry=apply_symmetry, apply_bg_color=adjust_bg_color, apply_hair_removal=remove_hairs)
            for i, img in enumerate(enhanced_images):
                st.image(img, caption=f"Enhanced Image {i+1}")

        image_indices = [i for i in range(len(enhanced_images))]
        votes = st.multiselect("Upvote the best images", image_indices)

        if st.button("Generate New Images"):
            if len(votes) > 0:
                selected_images = [enhanced_images[i] for i in votes]
                new_images = generate_new_images_based_on_feedback(selected_images)
                for i, img in enumerate(new_images):
                    st.image(img, caption=f"New Image {i+1}")
            else:
                st.warning("Please upvote at least one image before generating new ones.")
