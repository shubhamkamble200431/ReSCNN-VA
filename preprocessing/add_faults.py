import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Input/output paths
    input_path: str = "input.jpg"       # single image OR folder
    output_path: str = "output"         # output folder or file

    # Mode: "single" = process one image, "folder" = process all images in folder
    input_mode: str = "folder"          # "single" or "folder"

    # Noise options
    apply_gaussian: bool = True
    gaussian_mean: float = 0.0
    gaussian_std: float = 25.0  # Higher = stronger noise

    apply_impulse: bool = True
    impulse_amount: float = 0.02  # ratio of pixels to corrupt

    # Illumination options
    apply_illumination: bool = True
    illumination_factor: float = 1.2  # >1 = brighter, <1 = darker

    # Execution mode
    mode: str = "together"
    # options: "gaussian", "impulse", "illumination", "together"


def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_impulse_noise(image, amount=0.02):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

    # Salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


def change_illumination(image, factor=1.2):
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted


def process_image(image, config: Config):
    output = image.copy()

    if config.mode == "gaussian" and config.apply_gaussian:
        output = add_gaussian_noise(output, config.gaussian_mean, config.gaussian_std)

    elif config.mode == "impulse" and config.apply_impulse:
        output = add_impulse_noise(output, config.impulse_amount)

    elif config.mode == "illumination" and config.apply_illumination:
        output = change_illumination(output, config.illumination_factor)

    elif config.mode == "together":
        if config.apply_gaussian:
            output = add_gaussian_noise(output, config.gaussian_mean, config.gaussian_std)
        if config.apply_impulse:
            output = add_impulse_noise(output, config.impulse_amount)
        if config.apply_illumination:
            output = change_illumination(output, config.illumination_factor)

    return output


def run(config: Config):
    if config.input_mode == "single":
        # Process a single image
        image = cv2.imread(config.input_path)
        if image is None:
            raise ValueError("Image not found: " + config.input_path)

        output = process_image(image, config)

        # Save as single file
        if not config.output_path.lower().endswith((".jpg", ".png")):
            os.makedirs(config.output_path, exist_ok=True)
            save_path = os.path.join(config.output_path, "output.jpg")
        else:
            save_path = config.output_path

        cv2.imwrite(save_path, output)
        print(f"Processed single image saved to {save_path}")

    elif config.input_mode == "folder":
        # Process all images in a folder
        os.makedirs(config.output_path, exist_ok=True)
        for fname in os.listdir(config.input_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                img_path = os.path.join(config.input_path, fname)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                output = process_image(image, config)

                save_path = os.path.join(config.output_path, fname)
                cv2.imwrite(save_path, output)
                print(f"Saved {save_path}")

    else:
        raise ValueError("Invalid input_mode. Use 'single' or 'folder'.")


if __name__ == "__main__":
    # Example Config
    cfg = Config(
        input_path="/home/ml/Desktop/shubham/sensors/test_images",          # folder with images OR single file
        output_path="preprocess/25-0.01-2", # output folder
        input_mode="folder",          # "single" or "folder"
        apply_gaussian=True, gaussian_std=25,
        apply_impulse=True, impulse_amount=0.01,
        apply_illumination=True, illumination_factor=2,
        mode="together"               # "gaussian", "impulse", "illumination", "together"
    )

    run(cfg)
