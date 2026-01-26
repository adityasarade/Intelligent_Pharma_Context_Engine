import os
import random
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from preprocessing import preprocess_pipeline
from ocr import run_ocr, extract_text_blocks

def test_pipeline():
    # Find images from both datasets
    roboflow_images = glob.glob("data/raw/medicine bottle.v1i.yolov12/test/images/*.jpg")
    pills_images = glob.glob("data/raw/pills_inside_bottles/data/*.jpg") # Adjust path if needed based on actual structure
    
    all_images = roboflow_images + pills_images
    
    if not all_images:
        # Fallback search if paths are slightly different
        all_images = list(Path("data").rglob("*.jpg")) + list(Path("data").rglob("*.png"))
        all_images = [str(p) for p in all_images]
    
    if not all_images:
        print("No images found in data/ directory!")
        return

    # Select a random image
    sample_image = random.choice(all_images)
    print(f"Testing on image: {sample_image}")
    
    # 1. Run Preprocessing
    print("Running Preprocessing...")
    stages = preprocess_pipeline(sample_image)
    
    # 2. Run OCR on the enhanced image
    print("Running OCR...")
    ocr_text = run_ocr(stages['enhanced_gray'])
    print("\n--- Raw OCR Output ---")
    print(ocr_text)
    print("----------------------\n")
    
    blocks = extract_text_blocks(stages['enhanced_gray'])
    print(f"Found {len(blocks)} confident text blocks.")
    for b in blocks:
        print(f"  - [{b['conf']}%] {b['text']}")

    # 3. Visual Verification (Save plot since we can't show GUI)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(stages['original'], cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(stages['enhanced_gray'], cmap='gray')
    plt.title("Enhanced (CLAHE)")
    plt.axis("off")
    
    # Draw boxes on original
    vis_img = stages['original'].copy()
    for b in blocks:
        x1, y1, x2, y2 = b['bbox']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, str(b['conf']), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("OCR Detections")
    plt.axis("off")
    
    output_path = "docs/day1_test_result.png"
    plt.savefig(output_path)
    print(f"\nVisual result saved to {output_path}")

if __name__ == "__main__":
    test_pipeline()
