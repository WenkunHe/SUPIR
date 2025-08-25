import os
from PIL import Image
import json

files = os.listdir("00000049")

buckets = []
counts = []

for file in files:
    """
    if file.endswith(".json"):
        with open(os.path.join("00000049", file), "r") as json_file:
            meta = json.load(json_file)
        height, width = meta["height"], meta["width"]
        if (height, width) not in buckets:
            buckets.append((height, width))
            counts.append(1)
        else:
            idx = buckets.index((height, width))
            counts[idx] += 1
    """
    if file.endswith(".jpg"):
        img = Image.open(os.path.join("00000049", file))
        height, width = img.height, img.width
        if (height, width) not in buckets:
            buckets.append((height, width))
            counts.append(1)
        else:
            idx = buckets.index((height, width))
            counts[idx] += 1

for (bucket, count) in zip(buckets, counts):
    print(bucket, count)