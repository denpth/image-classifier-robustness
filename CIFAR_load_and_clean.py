'''
This script loads the CIFAR-10 dataset and performs some basic cleaning.
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# Global variables
IMG_SIZE = 32
NUM_CLASSES = 10
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
OUTPUT_FILE = "cifar10_cleaned.npz"
BLUR_THRESHOLD = 0
NOISE_THRESHOLD = 1000
ARTIFACT_THRESHOLD = 0.95

def load_cifar10():
    """
    Load the CIFAR-10 dataset.
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("CIFAR-10 dataset loaded.")
    return x_train, y_train, x_test, y_test

def clean_cifar10(x_train, y_train, x_test, y_test):
    """
    Clean the CIFAR-10 dataset.
    """
    print("Cleaning CIFAR-10 dataset...")
    print("Removing NaN values...")
    valid_mask = ~np.isnan(x_train).any(axis=(1, 2, 3))
    x_train = x_train[valid_mask]
    y_train = y_train[valid_mask]
    valid_mask = ~np.isnan(x_test).any(axis=(1, 2, 3))
    x_test = x_test[valid_mask]
    y_test = y_test[valid_mask]

    print("CIFAR-10 dataset cleaned.")
    return x_train, y_train, x_test, y_test

def normalize_cifar10(x_train, y_train, x_test, y_test):
    """
    Normalize the CIFAR-10 dataset.
    """
    print("Normalizing CIFAR-10 dataset...")
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print("CIFAR-10 dataset normalized.")
    return x_train, y_train, x_test, y_test

def one_hot_encode_cifar10(x_train, y_train, x_test, y_test):
    """
    One-hot encode the CIFAR-10 dataset.
    """
    print("One-hot encoding CIFAR-10 dataset...")
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    print("CIFAR-10 dataset one-hot encoded.")
    return x_train, y_train, x_test, y_test

def split_cifar10(x_train, y_train, x_test, y_test):
    """
    Split the CIFAR-10 dataset.
    """
    print("Splitting CIFAR-10 dataset...")
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_train)
    print("CIFAR-10 dataset split.")
    return x_train, y_train, x_val, y_val, x_test, y_test

def save_cifar10(x_train, y_train, x_test, y_test, x_val = None, y_val = None):
    """
    Save the CIFAR-10 dataset.
    """
    print("Saving CIFAR-10 dataset...")
    np.savez_compressed(OUTPUT_FILE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_val=x_val, y_val=y_val)
    print("CIFAR-10 dataset saved.")
    return 0

def display_stats(x_train, y_train, x_test, y_test, x_val = None, y_val = None):
    """
    Display the statistics of the CIFAR-10 dataset.
    """
    print("Displaying CIFAR-10 dataset statistics...")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    if x_val is not None and y_val is not None:
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
    print("CIFAR-10 dataset statistics displayed.")
    return 0

def display_images(x_train, y_train, x_test, y_test):
    """
    Display the images of the CIFAR-10 dataset.
    """
    print("Displaying CIFAR-10 dataset images...")
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()
    print("CIFAR-10 dataset images displayed.")
    return 0

def check_blur(image):
    """
    Check if the image is blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD

def check_noise(image):
    """
    Check if the image is noisy.
    """
    denoised = cv2.medianBlur(image, 3)
    diff = image.astype(np.float32) - denoised.astype(np.float32)
    mse = np.mean(diff ** 2)
    return mse > NOISE_THRESHOLD

def check_and_repair_artifacts(image):
    """
    Check if the image has artifacts.
    """
# Thresholds
    MAX_ARTIFACT_COVERAGE = 0.20  # If > 20% is watermark, delete image
    MIN_ARTIFACT_COVERAGE = 0.02  # If < 2% is edges, likely just texture (ignore)
    
    h, w = image.shape[:2]
    total_pixels = h * w
    
    # 1. Edge Detection (Find the artifact)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 200, 2000)
    
    # 2. Morphological Dilation (The "Blobbing" Trick)
    # Text is made of thin lines. We 'thicken' them to connect letters into a solid mask.
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    
    # 3. Calculate Coverage
    affected_pixels = np.count_nonzero(mask)
    coverage_ratio = affected_pixels / total_pixels
    
    # 4. Decision Logic
    
    # CASE A: Massive Watermark (e.g., Center of image)
    if coverage_ratio > MAX_ARTIFACT_COVERAGE:
        # Return None to signal "Discard this image"
        return None, "discard"
        
    # CASE B: Small/Localized Watermark (e.g., Corner or small tag)
    elif coverage_ratio > MIN_ARTIFACT_COVERAGE:
        # Repair the specific area defined by the mask
        # inpaint_radius=3 looks at pixels 3 steps away to fix the hole
        repaired_img = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return repaired_img, "repaired"
        
    # CASE C: Clean Image (Just natural texture)
    else:
        return image, "clean"

def pad_to_square(image):
    """
    Dynamic Letterboxing: Adds black borders to make image square 
    without stretching aspect ratio.
    """
    height, width = image.shape[:2]
    if height == width:
        return image
    
    # Calculate padding
    diff = abs(height - width)
    p1 = diff // 2
    p2 = diff - p1
    
    if width > height: # Landscape -> Pad Top/Bottom
        image = cv2.copyMakeBorder(image, p1, p2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:              # Portrait -> Pad Left/Right
        image = cv2.copyMakeBorder(image, 0, 0, p1, p2, cv2.BORDER_CONSTANT, value=[0,0,0])
    return image

def perform_quality_gate(x_train, y_train):
    """
    Perform quality gate on the CIFAR-10 dataset.
    """
    print("Performing quality gate on CIFAR-10 dataset...")
    cleaned_x_train = []
    cleaned_y_train = []

    stats = {
        'kept': 0,
        'repaired_artifacts': 0,
        'discarded_artifacts': 0,
        'discarded_noise': 0,
        'discarded_blur': 0,
        'padded': 0,
    }
        
    
    for i in tqdm(range(len(x_train)), desc="Inspecting images"):
        img = x_train[i]
        lbl = y_train[i]
        
        img, status = check_and_repair_artifacts(img)
        if status == "clean":
            stats['kept'] += 1
        elif status == "repaired":
            stats['repaired_artifacts'] += 1
        elif status == "discard":
            stats['discarded_artifacts'] += 1
            continue
        
        if check_noise(img):
            stats['discarded_noise'] += 1
            continue

        if check_blur(img):
            stats['discarded_blur'] += 1
            continue
            
        if img.shape[0] != img.shape[1]:
            img = pad_to_square(img)
            stats['padded'] += 1

        if img.shape[0] != 32 or img.shape[1] != 32:
            img = cv2.resize(img, (32, 32))
        
        cleaned_x_train.append(img)
        cleaned_y_train.append(lbl)
    
    print("Quality gate performed.")
    print("Stats:", stats)
    return np.array(cleaned_x_train), np.array(cleaned_y_train)
    
    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train, y_train, x_test, y_test = clean_cifar10(x_train, y_train, x_test, y_test)
    x_train, y_train = perform_quality_gate(x_train, y_train)
    x_train, y_train, x_test, y_test = normalize_cifar10(x_train, y_train, x_test, y_test)
    x_train, y_train, x_test, y_test = one_hot_encode_cifar10(x_train, y_train, x_test, y_test)
    x_train, y_train, x_val, y_val, x_test, y_test = split_cifar10(x_train, y_train, x_test, y_test)
    display_stats(x_train, y_train, x_test, y_test, x_val, y_val)
    display_images(x_train, y_train, x_test, y_test)

    save_cifar10(x_train, y_train, x_test, y_test, x_val, y_val)
