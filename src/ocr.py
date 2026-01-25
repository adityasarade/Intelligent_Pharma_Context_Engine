import pytesseract
import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Dict, Any, Union

# Set Tesseract command if not in PATH (Optional, handled by env var usually)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_tesseract_cmd():
    return os.getenv("TESSERACT_CMD", None)

if get_tesseract_cmd():
    pytesseract.pytesseract.tesseract_cmd = get_tesseract_cmd()

def run_ocr(image: Union[np.ndarray, Image.Image], lang: str = 'eng') -> str:
    """
    Runs Tesseract OCR on the given image.
    Returns the raw string output.
    """
    # Configuration:
    # --oem 3: Default, based on what causes the best results (LSTM usually)
    # --psm 3: Fully automatic page segmentation, but no OSD.
    # --psm 6: Assume a single uniform block of text (good for cropped labels)
    config = r'--oem 3 --psm 6' 
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text.strip()

def run_ocr_with_data(image: Union[np.ndarray, Image.Image], lang: str = 'eng') -> Dict[str, Any]:
    """
    Runs Tesseract and returns detailed data (bounding boxes, confidences).
    """
    config = r'--oem 3 --psm 6'
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    return data

def extract_text_blocks(image: np.ndarray, min_conf: int = 40) -> List[Dict[str, Any]]:
    """
    Runs OCR and filters for confident text blocks.
    Returns list of {text, conf, bbox, block_num, line_num}
    """
    data = run_ocr_with_data(image)
    blocks = []
    
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if conf > min_conf and len(text) > 1:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blocks.append({
                "text": text,
                "conf": conf,
                "bbox": (x, y, x+w, y+h), # x1, y1, x2, y2
                "block_num": data['block_num'][i],
                "line_num": data['line_num'][i]
            })
            
    return blocks
