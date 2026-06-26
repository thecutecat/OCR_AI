import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_business_card():
    """Create a synthetic business card image for testing."""
    # Create a white background
    img = Image.new('RGB', (1200, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a card border
    draw.rectangle([100, 100, 1100, 700], outline='black', width=3)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 60)
        font_small = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    draw.text((200, 200), "John Doe", fill='black', font=font)
    draw.text((200, 280), "Senior Software Engineer", fill='gray', font=font_small)
    draw.text((200, 360), "ABC Corporation", fill='black', font=font_small)
    draw.text((200, 440), "john.doe@abccorp.com", fill='blue', font=font_small)
    draw.text((200, 520), "+62-812-3456-7890", fill='black', font=font_small)
    
    # Add a logo placeholder
    draw.rectangle([850, 200, 950, 350], outline='gray', width=2)
    draw.text((860, 260), "Logo", fill='gray', font=font_small)
    
    return np.array(img)

def create_test_dataset():
    """Create a test dataset with various conditions."""
    # Create output directory
    os.makedirs("samples", exist_ok=True)
    
    # 1. Normal business card
    card = create_business_card()
    cv2.imwrite("samples/business_card_normal.jpg", cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
    print("Created: samples/business_card_normal.jpg")
    
    # 2. Rotated card
    card_rotated = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("samples/business_card_rotated.jpg", cv2.cvtColor(card_rotated, cv2.COLOR_RGB2BGR))
    print("Created: samples/business_card_rotated.jpg")
    
    # 3. Card with perspective distortion
    # Simulate perspective by applying a homography
    h, w = card.shape[:2]
    src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst_pts = np.float32([[50, 50], [w-100, 20], [w-80, h-80], [30, h-50]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    card_perspective = cv2.warpPerspective(card, M, (w, h))
    cv2.imwrite("samples/business_card_perspective.jpg", cv2.cvtColor(card_perspective, cv2.COLOR_RGB2BGR))
    print("Created: samples/business_card_perspective.jpg")
    
    # 4. Card with low light
    card_low_light = cv2.convertScaleAbs(card, alpha=0.3, beta=0)
    cv2.imwrite("samples/business_card_low_light.jpg", cv2.cvtColor(card_low_light, cv2.COLOR_RGB2BGR))
    print("Created: samples/business_card_low_light.jpg")
    
    # 5. Card with noise
    noise = np.random.normal(0, 30, card.shape).astype(np.uint8)
    card_noisy = cv2.add(card, noise)
    cv2.imwrite("samples/business_card_noisy.jpg", cv2.cvtColor(card_noisy, cv2.COLOR_RGB2BGR))
    print("Created: samples/business_card_noisy.jpg")
    
    print("\nTest dataset created successfully!")

if __name__ == "__main__":
    create_test_dataset()