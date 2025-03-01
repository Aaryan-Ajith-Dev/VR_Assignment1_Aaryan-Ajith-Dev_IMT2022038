# VR_Assignment1_Aaryan-Ajith-Dev_IMT2022038

## Part 1: Coin Segmentation

### Instructions:
1. Navigate to the `part1` folder.
2. The input image used is `coins.jpeg`.
3. The script `script1.py` generates the results.

### Functionality:
- The `part1()` function produces results for **Part A**.
- The `part2_3()` function generates results for **Part B** and **Part C**.
- By default, the script runs both functions sequentially.

### To Run:
```bash
cd part1
python3 script1.py
```

---

## Part 2: Image Panorama Stitching

### Instructions:
1. Navigate to the `part2` folder.
2. The input images used are `left.jpeg`, `middle.jpeg`, and `right.jpeg`.
3. The script `script2.py` is used to generate the panorama.

### Functionality:
- The `create_panorama()` function in `script2.py` stitches two images to create a panorama.
- To create a panorama from three images:
  1. **Step 1:** Stitch `middle.jpeg` and `right.jpeg` to produce `output.jpeg`.
  2. **Step 2:** Stitch `output.jpeg` with `left.jpeg` to generate the final `final.jpeg`.

### To Run:
```bash
cd part2
python3 script2.py
```

---

## Additional Dependencies

Ensure the following Python packages are installed:
```bash
pip install opencv-python numpy matplotlib
```
- `opencv-python` (cv2)
- `numpy`
- `matplotlib` (for visualization)

