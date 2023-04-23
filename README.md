# Image Uploader and Variations Generator

This is a Streamlit application that allows you to upload an image, apply various filters using sliders, resize the filtered image, and generate variations using the OpenAI API. The generated images are saved as PNG files.

![Demo](demo.png)

## Features

- Upload an image in PNG, JPG, or JPEG format
- Apply filters to the image using sliders in the sidebar:
  - Brightness
  - Contrast
  - Sharpness
  - Saturation
  - Rotation
- Resize the filtered image
- Generate variations using the OpenAI API
- Save generated images as PNG files

## Installation

1. Clone the Blue Haze repository or download the source code.

git clone https://github.com/bmszone42/blue-haze.git


2. Navigate to the project directory.

cd blue-haze


3. Add the Streamlit application (`image_app.py`) and the `requirements.txt` file to the project directory.

4. Install the required packages.

pip install -r requirements.txt

5. Run the Streamlit application.

streamlit run image_app.py


6. Open the application in your web browser using the URL displayed in the terminal.

## Usage

1. Enter your OpenAI API key in the text input field.
2. Upload an image using the file uploader.
3. Adjust the filter sliders in the sidebar to modify the image.
4. Set the number of variations you want to generate.
5. The original image, filtered image, and generated variations will be displayed in the main window.

## License

MIT License
