import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_dummy_data():
    output_dir = Path("data/test_ingest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Excel File
    excel_path = output_dir / "AMZN_10-K_0000320193-25-000079.xlsx"
    df = pd.DataFrame({
        'Item': ['Laptop', 'Mouse', 'Monitor'],
        'Quantity': [5, 10, 2],
        'Price': [1200, 25, 300]
    })
    df.to_excel(excel_path, index=False)
    print(f"Created {excel_path}")
    
    # 2. Image File
    # Create an image with text "Revenue Growth 2024: +15%"
    img_path = output_dir / "AMZN_10-K_0000320193-25-000079.png"
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        
    d.text((50, 80), "Revenue Growth 2024: +15%", fill=(0, 0, 0), font=font)
    img.save(img_path)
    print(f"Created {img_path}")
    
    return output_dir

if __name__ == "__main__":
    create_dummy_data()
