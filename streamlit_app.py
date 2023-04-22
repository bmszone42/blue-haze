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

def deblur_image(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def denoise_image(image, strength=10):
  return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

def upscale_image(image, factor=2):
    return cv2.resize(image, (image.shape[1]*factor, image.shape[0]*factor), interpolation=cv2.INTER_CUBIC)

def color_grade_image(image, color_balance=[1, 1, 1]):
  color_matrix = np.array([[color_balance[0], 0, 0],
                           [0, color_balance[1], 0],
                           [0, 0, color_balance[2]]])
  return cv2.transform(image, color_matrix)

def adjust_white_balance(image, temperature=0, tint=0):
    kelvin = 2735 - temperature
    matrix = cv2.get_color_temperature_correction(int(kelvin), tint)
    return cv2.transform(image, matrix)
  
def crop_to_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y+h, x:x+w]
    else:
        return image

def recompose_image(image):
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(image, None)
    if len(kp) < 4:
        return image
    else:
        # Find the convex hull of the keypoints and crop to its bounding box
        pts = np.float32([kp[idx].pt for idx in range(len(kp))]).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        x, y, w, h = cv2.boundingRect(hull)
        return image[y:y+h, x:x+w]

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
  
def color_correction(image, brightness=0, contrast=0, saturation=0, gamma=1):
  # Adjust brightness and contrast
  brightness = int((brightness - 0.5) * 255 * 2)
  contrast = int((contrast - 0.5) * 255 * 2)
  alpha = (255 + contrast) / 255
  beta = brightness

  corrected = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

  # Adjust saturation
  hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  s = np.clip(s * (1 + saturation), 0, 255)
  hsv = cv2.merge([h, s, v])
  corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

  # Apply gamma correction
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
  corrected = cv2.LUT(corrected, table)

  return corrected

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = image[y:y+h, x:x+w]
    return cropped

def create_collage(images):
    images = [image.resize((int(image.width/2), int(image.height/2))) for image in images]  # Resize the images
    rows = [np.concatenate(images[i:i+4], axis=1) for i in range(0, 12, 4)]  # Concatenate the images into rows
    collage = np.concatenate(rows, axis=0)  # Concatenate the rows into a single image
    return Image.fromarray(collage)

def apply_improvements(image, apply_lighting=False, apply_symmetry=False, apply_bg_color=False, apply_hair_removal=False, adjust_color=False, auto_crop=False):
    if apply_lighting:
        image = improve_lighting(image)
    if apply_symmetry:
        image = enhance_symmetry(image)
    if apply_bg_color:
        image = adjust_background_color(image)
    if apply_hair_removal:
        image = remove_stray_hairs(image)
    if adjust_color:
        image = color_correction(image, brightness=brightness, contrast=contrast, saturation=saturation, gamma=gamma)
    if auto_crop:
        image = auto_crop(image)

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

def apply_improvements(image, 
                       apply_lighting=False, 
                       apply_symmetry=False, 
                       apply_bg_color=False, 
                       apply_hair_removal=False,
                       apply_deblurring=False,
                       apply_denoising=False,
                       apply_superresolution=False,
                       apply_color_grading=False,
                       apply_color_correction=False,
                       apply_auto_crop=False,
                       apply_auto_compose=False):
    
    if apply_deblurring:
        image = deblur(image)
    
    if apply_denoising:
        image = denoise(image)
        
    if apply_superresolution:
        image = superresolve(image)
    
    if apply_lighting:
        image = improve_lighting(image)
        
    if apply_symmetry:
        image = enhance_symmetry(image)
        
    if apply_bg_color:
        image = adjust_background_color(image)
        
    if apply_hair_removal:
        image = remove_stray_hairs(image)
        
    if apply_color_grading:
        image = apply_color_grade(image)
        
    if apply_color_correction:
        image = apply_color_correction(image)
        
    if apply_auto_crop:
        image = auto_crop(image)
        
    if apply_auto_compose:
        image = auto_compose(image)
        
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
            deblur_image = st.sidebar.checkbox("Deblur Image", help="Remove motion blur from the image")
            denoise_image = st.sidebar.checkbox("Denoise Image", help="Reduce noise in the image")
            superresolve_image = st.sidebar.checkbox("Superresolve Image", help="Increase resolution of the image")
            color_grade = st.sidebar.checkbox("Color Grade", help="Adjust the overall color balance of the image")
            color_correction = st.sidebar.checkbox("Color Correction", help="Correct the color cast of the image")
            auto_cropper = st.sidebar.checkbox("Auto Crop", help="Automatically crop the image to the most visually interesting part")
            auto_composer = st.sidebar.checkbox("Auto Compose", help="Automatically recompose the image for a more compelling composition")

            if enhance_lighting:
                selected_enhancements.append("improve_lighting")
            if enhance_symmetry:
                selected_enhancements.append("enhance_symmetry")
            if adjust_bg_color:
                selected_enhancements.append("adjust_background_color")
            if remove_hairs:
                selected_enhancements.append("remove_stray_hairs")
            if deblur_image:
                selected_enhancements.append("deblur_image")
            if denoise_image:
                selected_enhancements.append("denoise_image")
            if superresolve_image:
                selected_enhancements.append("superresolve_image")
            if color_grade:
                selected_enhancements.append("color_grade")
            if color_correction:
                selected_enhancements.append("color_correction")
            if auto_cropper:
                selected_enhancements.append("auto_crop")
            if auto_composer:
                selected_enhancements.append("auto_compose")

            enhanced_images = []
            if st.button("Generate Enhanced Images"):
                for i in range(12):
                    enhanced_image = apply_improvements(np.array(input_image), 
                                                         apply_lighting=enhance_lighting, 
                                                         apply_symmetry=enhance_symmetry, 
                                                         apply_bg_color=adjust_bg_color, 
                                                         apply_hair_removal=remove_hairs,
                                                         apply_deblur=deblur_image,
                                                         apply_denoise=denoise_image,
                                                         apply_superresolution=superresolve_image,
                                                         apply_color_grade=color_grade,
                                                         apply_color_correction=color_correction,
                                                         apply_auto_crop=auto_crop,
                                                         apply_auto_compose=auto_compose)
                    enhanced_images.append(enhanced_image)
                    # Save the enhanced image to disk
                   


if __name__ == '__main__':
    main()
