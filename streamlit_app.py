import streamlit as st
import cv2
import numpy as np
from PIL import Image


st.set_page_config(page_title="PicPerfector: Ultimate Photo Transformation",
                   page_icon=":camera_flash:",
                   layout="wide")
st.markdown("<h1 style='text-align: center; color: #b89b7b'>PicPerfector: Ultimate Photo Transformation</h1>", 
            unsafe_allow_html=True)

selected_enhancements = []

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

def create_collage(images):
    images = [image.resize((int(image.width/2), int(image.height/2))) for image in images]  # Resize the images
    rows = [np.concatenate(images[i:i+4], axis=1) for i in range(0, 12, 4)]  # Concatenate the images into rows
    collage = np.concatenate(rows, axis=0)  # Concatenate the rows into a single image
    return Image.fromarray(collage)

def apply_improvements(image, apply_lighting=False, apply_symmetry=False, apply_bg_color=False, apply_hair_removal=False):
    if apply_lighting:
        image = improve_lighting(image)
    if apply_symmetry:
        image = enhance_symmetry(image)
    if apply_bg_color:
        image = adjust_background_color(image)
    if apply_hair_removal:
        image = remove_stray_hairs(image)
    return image

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

def create_collage(images):
    images = [image.resize((int(image.width/2), int(image.height/2))) for image in images]  # Resize the images
    rows = [np.concatenate(images[i:i+4], axis=1) for i in range(0, 12, 4)]  # Concatenate the images into rows
    collage = np.concatenate(rows, axis=0)  # Concatenate the rows into a single image
    return Image.fromarray(collage)

def apply_improvements(image, apply_lighting=False, apply_symmetry=False, apply_bg_color=False, apply_hair_removal=False):
    if apply_lighting:
        image = improve_lighting(image)
    if apply_symmetry:
        image = enhance_symmetry(image)
    if apply_bg_color:
        image = adjust_background_color(image)
    if apply_hair_removal:
        image = remove_stray_hairs(image)
    return image

def select_and_save_image(images):
  # Display the generated images
  for i, img in enumerate(images):
      st.image(img, caption=f"Generated Image {i+1}")

  # Allow the user to select their favorite image
  selected_index = st.selectbox("Select your favorite image:", range(len(images)))

  # Save the selected image locally as a PNG file
  selected_image = Image.fromarray(images[selected_index])
  selected_image.save("favorite_image.png")

  st.success("Image saved successfully!")

def main():

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

            enhance_lighting = st.sidebar.checkbox("Improve Lighting", help="Enhance brightness and contrast of the image")
            enhance_symmetry = st.sidebar.checkbox("Enhance Facial Symmetry", help="Improve facial symmetry using reflection")
            adjust_bg_color = st.sidebar.checkbox("Adjust Background Color", help="Change the background color of the image")
            remove_hairs = st.sidebar.checkbox("Remove Stray Hairs", help="Remove unwanted hairs from the image")

            if enhance_lighting:
                selected_enhancements.append("improve_lighting")
            if enhance_symmetry:
                selected_enhancements.append("enhance_symmetry")
            if adjust_bg_color:
                selected_enhancements.append("adjust_background_color")
            if remove_hairs:
                selected_enhancements.append("remove_stray_hairs")

            num_images = st.slider("Number of images to generate", 1, 20, 12)

            enhanced_images = []
            if st.button("Generate Enhanced Images"):
                for i in range(num_images):
                    enhanced_image = apply_improvements(np.array(input_image), 
                                                         apply_lighting=enhance_lighting, 
                                                         apply_symmetry=enhance_symmetry, 
                                                         apply_bg_color=adjust_bg_color, 
                                                         apply_hair_removal=remove_hairs)
                    enhanced_images.append(enhanced_image)
                    # Save the enhanced image to disk
                    Image.fromarray(enhanced_image).save(f'image{i}_enhanced.png')
                    st.image(enhanced_image, caption=f"Enhanced Image {i+1}")

            selected_indices = st.multiselect("Upvote the best images",
                                  options=list(range(12)),
                                  format_func=lambda i: f"Enhanced Image {i+1}",
                                  default=[])

            if st.button("Generate New Images Based on Voting"):
                if len(selected_indices) > 0:
                    selected_images = [enhanced_images[i] for i in selected_indices]
                    new_images = generate_new_images_based_on_feedback(selected_images)
                    collage = create_collage(new_images)
                    st.image(collage, caption='Enhanced Images Collage')
                else:
                    st.warning("Please upvote at least one image before generating new images")


if __name__ == '__main__':
    main()
