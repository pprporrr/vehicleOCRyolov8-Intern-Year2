from ultralytics import YOLO
import torch
import os
from PIL import Image

model = YOLO('Model/plateOCR.pt')
classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ']

def predict(imgPath):
    img = Image.open(imgPath)
    
    result = model(img)
    boxes = result[0].boxes.data
    sortedBoxes = boxes[torch.argsort(boxes[:, 0])]
    lastIndices = sortedBoxes[:, -1].tolist()
    mappedClasses = [classNames[int(cls)] for cls in lastIndices]
    
    return mappedClasses

image_directory = '/Users/ppr/Desktop/Project/plateOCR/new/'

for filename in os.listdir(image_directory):
    if filename.endswith(".jpeg"):
        img_path = os.path.join(image_directory, filename)
        
        result = predict(img_path)
        
        print("Image: ", filename)
        print("Result: ", result)
        print()