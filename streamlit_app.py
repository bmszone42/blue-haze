import streamlit as st
from io import BytesIO
from PIL import Image, ImageEnhance
import requests
import os
import openai

# Function to generate variations
def generate_variations(image, n=1):
    variations = []

    try:
        # Convert the image to a BytesIO object
        byte_stream = BytesIO()
        image.save(byte_stream, format='PNG')
        byte_array = byte_stream.getvalue()

        # Generate image variations using OpenAI API
        response = openai.Image.create_variation(
            image=byte_array,
            n=n,
            size="1024x1024"
        )

        for data in response['data']:
            url = data['url']
            resp = requests.get(url)
            variation = Image.open(BytesIO(resp.content))
            variations.append(variation)

    except openai.error.OpenAIError as e:
        print(e.http_status)
        print(e.error)

    return variations

# Function to apply filters to the image
def apply_filters(image, brightness, contrast, sharpness, saturation, rotate):
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)

    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(contrast)

    enhancer_sharpness = ImageEnhance.Sharpness(image)
    image = enhancer_sharpness.enhance(sharpness)

    enhancer_saturation = ImageEnhance.Color(image)
    image = enhancer_saturation.enhance(saturation)

    image = image.rotate(rotate)

    return image

# Main application
def main():
    st.title("Image Uploader and Variations Generator")

    # API key input widget
    api_key = st.text_input("Enter your OpenAI API key:")

    if api_key:
        openai.api_key = api_key

        # File uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Read the uploaded image
            image = Image.open(uploaded_file)

            # Sidebar sliders for filters
            st.sidebar.title("Image Filters")
            brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
            contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
            sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)
            saturation = st.sidebar.slider("Saturation", 0.5, 3.0, 1.0)
            rotate = st.sidebar.slider("Rotate", 0, 360, 0)

            # Apply filters to the image
            filtered_image = apply_filters(image, brightness, contrast, sharpness, saturation, rotate)

            # Display the original and filtered images
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

            st.subheader("Filtered Image")
            st.image(filtered_image, use_column_width=True)

            # Resize the filtered image
            width, height = 256, 256
            filtered_image = filtered_image.resize((width, height))

            # Generate image variations
            n_variations = st.slider("Number of variations", 1, 10, 1)
            st.write(f"Generating {n_variations} variations...")

            variations = generate_variations(filtered_image, n_variations)

            # Display the generated variations
            st.subheader("Generated Variations")
            for i, variation in enumerate(variations):
                st.write(f"Variation {i+1}")
                st.image(variation, use_column_width=True)

                # Save the generated image as a PNG file
                output_file = f"generated_variation_{i+1}.png"
                variation.save(output_file)
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

if __name__ == "__main__":
    main()

