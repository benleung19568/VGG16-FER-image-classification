import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from PIL import Image, ImageEnhance, ImageOps
import shutil

def create_augmented_dataset(source_dir, output_dir, emotion=None, augmentation_factor=2):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    has_emotion_folders = False
    potential_emotions = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    # Folders match with emotion names
    common_emotions = ['happy', 'sad', 'fear', 'angry', 'neutral', 'surprise', 'disgust']
    emotion_folders = [f for f in potential_emotions if f.lower() in common_emotions]
    
    if emotion_folders:
        has_emotion_folders = True
        print(f"Found emotion folders: {emotion_folders}")
    else:
        print(f"No emotion subfolders found in {source_dir}. Processing all images in this directory.")
    
    # Process based on directory structure
    if has_emotion_folders:
        emotions = emotion_folders
        
        # Filter to specific emotion if provided
        if emotion and emotion in emotions:
            emotions = [emotion]
            
        # Process each emotion folder
        for emotion_folder in emotions:
            emotion_source_dir = os.path.join(source_dir, emotion_folder)
            emotion_output_dir = os.path.join(output_dir, emotion_folder)
            
            if not os.path.exists(emotion_output_dir):
                os.makedirs(emotion_output_dir)
            else:
                shutil.rmtree(emotion_output_dir)
                os.makedirs(emotion_output_dir)
            
            print(f"Copying original {emotion_folder} images...")
            for filename in tqdm(os.listdir(emotion_source_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(emotion_source_dir, filename)
                    dst_path = os.path.join(emotion_output_dir, filename)
                    shutil.copy2(src_path, dst_path)
            
            # Generate augmented images
            print(f"Generating augmented {emotion_folder} images...")
            for filename in tqdm(os.listdir(emotion_source_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    process_and_augment_image(emotion_source_dir, emotion_output_dir, filename, augmentation_factor)
            
            print(f"Augmentation for {emotion_folder} complete. Augmented images saved to {emotion_output_dir}")
    else:
        # Process all images in the source directory
        print("Copying original images...")
        for filename in tqdm(os.listdir(source_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy2(src_path, dst_path)
        
        # Generate augmented images
        print("Generating augmented images...")
        for filename in tqdm(os.listdir(source_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_and_augment_image(source_dir, output_dir, filename, augmentation_factor)
        
        print(f"Augmentation complete. Augmented images saved to {output_dir}")

def process_and_augment_image(source_dir, output_dir, filename, augmentation_factor):

    img_path = os.path.join(source_dir, filename)
    name, ext = os.path.splitext(filename)
    
    try:
        # Load image with PIL for some augmentations
        pil_img = Image.open(img_path)
        
        # Load image with OpenCV for other augmentations
        cv_img = cv2.imread(img_path)
        
        if cv_img is None:
            print(f"Warning: Could not load {filename} with OpenCV. Skipping.")
            return
            
        for i in range(augmentation_factor):
            # Apply random augmentations
            aug_img = apply_random_augmentations(pil_img, cv_img)
            
            # Save the augmented image
            aug_filename = f"{name}_aug_{i}{ext}"
            aug_path = os.path.join(output_dir, aug_filename)
            
            # Convert from BGR to RGB if needed and save
            if isinstance(aug_img, np.ndarray):
                cv2.imwrite(aug_path, aug_img)
            else:
                aug_img.save(aug_path)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

def apply_random_augmentations(pil_img, cv_img):
    
    # Choose whether to use PIL or OpenCV for this augmentation
    use_pil = random.choice([True, False])
    
    if use_pil:
        img = pil_img.copy()
        
        # Apply random PIL-based augmentations
        aug_type = random.randint(0, 4)
        
        if aug_type == 0:
            # Brightness adjustment
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        elif aug_type == 1:
            # Contrast adjustment
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        elif aug_type == 2:
            # Horizontal flip
            img = ImageOps.mirror(img)
        
        elif aug_type == 3:
            # Rotation (small angles to preserve facial features)
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        elif aug_type == 4:
            # Sharpness adjustment
            factor = random.uniform(0.7, 1.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)
        
        return img
    
    else:
        img = cv_img.copy()
        
        # Apply random OpenCV-based augmentations
        aug_type = random.randint(0, 4)
        
        if aug_type == 0:
            # Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        elif aug_type == 1:
            # Rotation with scaling
            angle = random.uniform(-15, 15)
            height, width = img.shape[:2]
            center = (width/2, height/2)
            scale = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, M, (width, height))
        
        elif aug_type == 2:
            # Horizontal flip
            img = cv2.flip(img, 1)
        
        elif aug_type == 3:
            # Add slight noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        elif aug_type == 4:
            # Slight perspective transform
            height, width = img.shape[:2]
            
            # Define the 4 source points
            pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            
            # Define the 4 destination points (slightly modified)
            shift = width * 0.05
            pts2 = np.float32([
                [0 + random.uniform(0, shift), 0 + random.uniform(0, shift)],
                [width - random.uniform(0, shift), 0 + random.uniform(0, shift)],
                [0 + random.uniform(0, shift), height - random.uniform(0, shift)],
                [width - random.uniform(0, shift), height - random.uniform(0, shift)]
            ])
            
            # Get transformation matrix
            M = cv2.getPerspectiveTransform(pts1, pts2)
            
            # Apply transformation
            img = cv2.warpPerspective(img, M, (width, height))
        
        return img

def main():
    source_dir = "C:/Users/user/Desktop/COMP7250_ML_MiniProject/SelfSyntheticFEDB/full"
    output_dir = "C:/Users/user/Desktop/COMP7250_ML_MiniProject/SelfSyntheticFEDB/Augmented"

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        source_dir = input("Please enter the correct path to your dataset: ")
        if not os.path.exists(source_dir):
            print("Path still doesn't exist. Exiting program.")
            return

    print(f"Contents of {source_dir}:")
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}/ (directory)")
        else:
            print(f"  - {item} (file)")
    
    try:
        aug_factor = int(input("Enter the number of augmented versions to create per image (default 5): "))
    except ValueError:
        print("Using default augmentation factor of 5.")
        aug_factor = 2
    
    # Create augmented dataset
    create_augmented_dataset(source_dir, output_dir, emotion=None, augmentation_factor=aug_factor)
    
    print("Augmentation process completed!")

if __name__ == "__main__":
    main()

