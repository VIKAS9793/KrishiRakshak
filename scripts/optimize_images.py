import os
from pathlib import Path
from PIL import Image

def optimize_image(input_path, output_path, max_size=(800, 800), quality=85):
    """Optimize an image by resizing and compressing."""
    img = Image.open(input_path)
    
    # Convert to RGB if RGBA and has transparency
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    
    # Resize while maintaining aspect ratio
    img.thumbnail(max_size, Image.LANCZOS)
    
    # Save with optimization
    output_format = 'JPEG' if str(output_path).lower().endswith('.jpg') else 'PNG'
    img.save(
        output_path,
        output_format,
        optimize=True,
        quality=quality
    )
    
    original_size = os.path.getsize(input_path) / 1024  # KB
    new_size = os.path.getsize(output_path) / 1024  # KB
    
    print(f"Optimized: {input_path} ({original_size:.1f}KB -> {new_size:.1f}KB, {((original_size - new_size) / original_size * 100):.1f}% reduction)")
    return output_path

def main():
    # Create optimized directory if it doesn't exist
    optimized_dir = Path("assets/optimized")
    optimized_dir.mkdir(exist_ok=True)
    
    # Optimize logo
    logo_path = "assets/logos/logo.png"
    optimized_logo = optimize_image(
        logo_path,
        optimized_dir / "logo.jpg",
        max_size=(200, 200),
        quality=90
    )
    
    # Optimize banner
    banner_path = "assets/banners/banner.png"
    optimized_banner = optimize_image(
        banner_path,
        optimized_dir / "banner.jpg",
        max_size=(1200, 400),
        quality=85
    )
    
    print("\nOptimization complete!")
    print(f"Optimized logo saved to: {optimized_logo}")
    print(f"Optimized banner saved to: {optimized_banner}")

if __name__ == "__main__":
    main()
