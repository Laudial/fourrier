import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from numpy.fft import fft2, fftshift, ifft2
import os
from datetime import datetime

def load_and_display_image(image_path):
    image = io.imread(image_path)
    return image

def resize_and_convert_to_gray(image):
    gray_image = color.rgb2gray(image)
    return gray_image

def calculate_and_display_fft(image):
    fft_result = fft2(image)
    magnitude_spectrum = np.abs(fftshift(fft_result))
    return magnitude_spectrum

def inverse_fft_and_display_image(fft_result):
    inverse_fft_result = ifft2(fft_result).real
    return inverse_fft_result

def optimize_fft_display(fft_result):
    magnitude_spectrum_shifted = np.abs(fftshift(fft_result))
    return magnitude_spectrum_shifted

# Dossier d'entrée et de sortie
input_folder = "./data/input"
output_folder_base = "./data/output"

# Traitement pour chaque image dans le dossier d'entrée
for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image_name, image_extension = os.path.splitext(filename)

        # Charger l'image
        original_image = load_and_display_image(image_path)

        # Créer un dossier pour chaque image
        output_folder = os.path.join(output_folder_base, f"{image_name}")
        os.makedirs(output_folder, exist_ok=True)

        # Afficher et sauvegarder chaque étape
        plt.figure(figsize=(15, 10))
        plt.suptitle("Étapes de traitement")

        plt.subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.title("Image originale")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        gray_image = resize_and_convert_to_gray(original_image)
        plt.imshow(gray_image, cmap='gray')
        plt.title("Image redimensionnée et en niveau de gris")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        fft_result = calculate_and_display_fft(gray_image)
        plt.imshow(np.log1p(fft_result), cmap='gray')
        plt.title("FFT 2D - Module")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        inverse_fft_result = inverse_fft_and_display_image(fft_result)
        plt.imshow(inverse_fft_result, cmap='gray')
        plt.title("Image obtenue par transformée inverse")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        magnitude_spectrum_shifted = optimize_fft_display(fft_result)
        plt.imshow(magnitude_spectrum_shifted, cmap='gray')
        plt.title("FFT 2D - Module avec fftshift")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(np.log1p(magnitude_spectrum_shifted), cmap='gray')
        plt.title("FFT 2D - Module avec fftshift et échelle logarithmique")
        plt.axis('off')


        # Sauvegarder la figure dans le dossier de sortie
        current_datetime = datetime.now().strftime("%Y-%m-%d_%HH %MMin %SSec")
        plt.savefig(os.path.join(output_folder, f'result_{current_datetime}.png'))
        plt.close()
