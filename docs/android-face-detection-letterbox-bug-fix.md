> From: https://chat.deepseek.com/a/chat/s/43f8e85f-41d4-4e98-bfaf-af8203da0a52

# you asked

Please explain this bug and the corresponding fix in simple terms

5. Major Bug Fix: Achieving Android/Python Parity on Non-Square Images (Aspect Ratio & Letterbox Decoding)
The Discrepancy
In macOS prototype runs (reference_pipeline.py), the DBSCAN clustering of the same 17 father-son photos yielded excellent results: 1 father cluster (13 photos) and 2 father-son clusters (2 photos each). However, on Android, the same photos resulted in subpar, highly fractured clusters: 4 tiny clusters of only 2 photos each, with most images treated as outlier noise.

A deep-dive parity audit comparing the bounding boxes and landmark coordinates printed in Android logcat.txt against Python report.json for portrait image IMG-20210506-WA0000.jpg (decoded to $719 \times 1280$) exposed the root cause:

Python Bounding Box left: 123.88 pixels.
Android Bounding Box left: 226.50 pixels (a huge error of $> 100$ pixels!).
Root Cause Analysis
The core of the issue was a math error in un-letterbox coordinate un-projection within the Android detectors.

1. Image Pre-processing (Both Parity)
To feed a non-square bitmap into the square BlazeFace input size ($192 \times 192$ or $128 \times 128$), the pipelines scale the bitmap preserving aspect ratio, then center it by padding the outer edges: $$\text{scale} = \min\left(\frac{192}{sourceWidth}, \frac{192}{sourceHeight}\right)$$ $$\text{padX} = \frac{192 - \text{scaledWidth}}{2}, \quad \text{padY} = \frac{192 - \text{scaledHeight}}{2}$$

2. The Android Math Bug (Un-projection)
To translate the detected bounding box $(cx, cy, w, h)$ and landmark coords from the normalized $192 \times 192$ space back to the source image space:

The Correct Formula (Python): Shift the normalized coordinate by the letterbox padding, then divide by the aspect scale factor: $$\text{Coordinate}{source} = \frac{\text{Coordinate}{input} - \text{pad}}{\text{scale}}$$
The Flawed Android Formula: Completely ignored the aspect-ratio padding and uniform scale! Instead, it performed a simple linear squashing (assuming the image was stretched to 1:1, not letterboxed): $$sx = \frac{sourceWidth}{192}, \quad sy = \frac{sourceHeight}{192}$$ $$\text{Coordinate}{source} = \text{Coordinate}{input} \times sx$$

---

# deepseek response

Imagine you have a wide landscape photo, but the face detector model only accepts square images. To make it fit, you shrink the photo until its longer side fits the square, then pad the shorter sides with black bars (letterboxing). The detector sees the face inside this padded square. Later, you need to map the face’s position back to the original photo.

**The bug** was in the “mapping back” math on Android. It treated the input as if the original photo had simply been squashed or stretched to fill the square completely, ignoring the black bars. Instead of shifting coordinates to account for the black bars and then undoing the uniform scale, it just linearly stretched coordinates from the square to the photo’s dimensions. For a portrait photo (719 × 1280), the square had huge horizontal black bars, so the detected face appeared shifted to the right inside the square. Android’s formula didn’t subtract that bar width, so the face coordinates ended up offset by over 100 pixels—like measuring a person’s position on a map but forgetting they had been placed on a giant empty margin first.

**The fix** uses the correct formula that the Python pipeline already had: first slide the coordinate left/up by the width of the black bars, then divide by the single scaling factor used to fit the image. This undoes the letterbox step exactly, giving the true position on the original photo. With accurate coordinates, the clustering algorithm could correctly group face photos into father and father-son clusters, matching the Python results.

---

# you asked

This ought to be a common pattern in computer vision and image processing, isn't it?

---

# deepseek response

Yes, absolutely. This is a **textbook common pitfall** in computer vision pipelines, especially when adapting models that require fixed‑size square inputs (like many face detectors, object detectors, and classifiers).

The pattern is:

1. **Input image** is non‑square (e.g., 1920×1080, 719×1280).  
2. **Preprocessing** resizes keeping aspect ratio and then **letterboxes** (pads with black bars) to make a square tensor (e.g., 192×192).  
3. The model detects or predicts in that square space.  
4. You then need to **map coordinates back** to the original image.

The bug appears when you forget that the model’s coordinate system is on the *padded* image, not on a simple stretched version. A naive linear rescale (multiply by `orig_width/input_width` and `orig_height/input_height`) is wrong because the letterboxing introduced offsets (the black bars). The correct recovery must:

* **Subtract the padding offset** first, then **divide by the uniform scale factor** (or, equivalently, work in normalized coordinates relative to the padded image and then un‑pad).

Why it’s so common:

* **Many tutorials, code examples, and even some frameworks’ post‑processing utilities** fail to handle letterbox properly, leading to a recurring source of “off‑by‑margin” errors.  
* Developers often test only with square or near‑square images, where the padding is tiny and the error is negligible, so the bug goes unnoticed until a strongly non‑square image (like a tall portrait photo) appears.  
* The incorrect linear mapping intuitively feels right (“just stretch back”) to someone who hasn’t walked through the exact math.

In practice, you’ll see this bug pattern in hobby projects, Kaggle notebooks, and even production mobile apps—exactly as you encountered. That’s why seasoned vision engineers almost always write a dedicated `undo_letterbox(bboxes, pad, scale)` utility and unit‑test it with extreme aspect ratios (like 1:10 or 10:1) to surface the issue immediately.