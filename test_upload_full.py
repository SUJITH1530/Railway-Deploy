#!/usr/bin/env python
from app import app
import os
import json

# Create test image
test_img_path = 'test_img.jpg'

try:
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (100, 100), color='red')
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill='blue')
    img.save(test_img_path, 'JPEG')
    print("✅ Created test image using PIL")
except:
    with open(test_img_path, 'wb') as f:
        jpeg_data = bytes([
            0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d, 0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12,
            0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0xff, 0xc0, 0x00, 0x0b,
            0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x1f,
            0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
            0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00,
            0x00, 0x3f, 0x00, 0xfb, 0xd0, 0xff, 0xd9
        ])
        f.write(jpeg_data)
    print("✅ Created minimal test JPEG")

try:
    with app.test_client() as client:
        with open(test_img_path, 'rb') as f:
            resp = client.post('/upload', 
                data={'image': (f, 'test.jpg'), 'lane': 'lane1'},
                content_type='multipart/form-data')
        
        print(f'\nResponse status: {resp.status_code}')
        
        if resp.status_code == 200:
            data = resp.get_json()
            print('✅ Upload SUCCESS')
            print(f"\nFull response keys: {list(data.keys())}")
            print(f"\nprocessed_image: {data.get('processed_image')}")
            print(f"vehicles: {data.get('vehicles')}")
            print(f"green_time: {data.get('green_time')}")
            print(f"vehicle_types: {data.get('vehicle_types')}")
            
            # Check if file exists
            img_file = data.get('processed_image', '').lstrip('/')
            img_path = os.path.join(os.path.dirname(__file__), img_file)
            if os.path.exists(img_path):
                print(f"✅ Processed image file EXISTS: {img_path}")
            else:
                print(f"❌ Processed image file NOT FOUND: {img_path}")
                # List what's actually in the processed directory
                processed_dir = os.path.join(os.path.dirname(__file__), 'static', 'processed')
                if os.path.exists(processed_dir):
                    files = os.listdir(processed_dir)
                    print(f"   Files in static/processed: {files}")
        else:
            print(f'❌ Upload FAILED with status {resp.status_code}')
            print('Response:', resp.data.decode()[:500])
            
finally:
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
