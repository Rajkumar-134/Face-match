import csv
import cv2
from sklearn.metrics import f1_score
from insightface_matcher_pretrained import InsightFaceMatcher
import os

CSV_FILE = "combined_test.csv"   # CSV file in the same directory as this script

BASE_DIR = r'C:\Users\Admin\OneDrive\Documents\Face_match\LFW'

matcher = InsightFaceMatcher(threshold=0.4)

pairs = []
with open(CSV_FILE, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        pairs.append(row)

y_true = []
y_pred = []

for idx, row in enumerate(pairs):
    img1_path, img2_path, is_same = row
    img1 = cv2.imread(os.path.join(BASE_DIR, img1_path))
    img2 = cv2.imread(os.path.join(BASE_DIR, img2_path))
    if img1 is None or img2 is None:
        print("Missing image:", os.path.join(BASE_DIR, img1_path), os.path.join(BASE_DIR, img2_path))
        continue
    result, similarity = matcher.compare(img1, img2)
    y_true.append(int(is_same))
    y_pred.append(1 if similarity > 0.4 else 0)
    if idx % 10 == 0:
        print(f"Processed {idx+1}/{len(pairs)} pairs")

print("Number of pairs processed:", len(y_true))
print("y_true distribution:", {v: y_true.count(v) for v in set(y_true)})
print("y_pred distribution:", {v: y_pred.count(v) for v in set(y_pred)})

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}") 