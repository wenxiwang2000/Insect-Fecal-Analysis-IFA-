# 🪲 Insect Fecal Analysis (IFA)
### A T.U.R.D.-based fecal deposit detection tool with adjustable sensitivity

**Insect Fecal Analysis (IFA)** is a Python-based application that extends the classical **T.U.R.D. fecal detection framework** with a key methodological improvement: **explicit user-adjustable detection sensitivity**.

In the original T.U.R.D. workflow, detection sensitivity is implicitly determined by fixed adaptive threshold parameters and cannot be directly tuned by the user. This limitation can reduce detection performance when analyzing datasets with variable imaging conditions or very small and faint deposits.

IFA introduces a **user-controlled sensitivity parameter** that directly modulates the local contrast comparison between each pixel and its surrounding neighborhood. By exposing sensitivity as an adjustable parameter, the system allows users to dynamically refine detection thresholds and achieve more robust identification of fecal deposits across diverse experimental setups.

This modification enables reliable detection of **small, low-contrast deposits** across different insect species while preserving the **core detection logic and measurement principles of the original T.U.R.D. method**.

---

# ✨ Key Features

IFA implements a complete **image-analysis pipeline for fecal deposit detection and quantification**, including:

- Grayscale conversion
- Adaptive thresholding
- Morphological cleanup
- Contour-based measurement

Detected deposits are quantified using multiple geometric metrics:

- **Area**
- **Perimeter**
- **Circularity**

The software also computes key biological metrics such as:

- **ROD classification**
- **IOD (Integrated Optical Density)**

---

# 🧠 Novelty and Improvements

Compared with the original **T.U.R.D. software**, IFA introduces several improvements:

### Adjustable detection sensitivity
A new **sensitivity parameter** allows direct control of adaptive threshold behavior, enabling improved detection of faint deposits.

### Interactive GUI
IFA provides an **interactive graphical interface** that allows users to:

- select regions of interest (ROI)
- adjust detection sensitivity
- visualize segmentation results in real time

### Optional background exclusion
False detections caused by background patterns can be reduced using:

- **HSV color masking**
- preset background filters
- manual background sampling within the ROI

### Arena masking
Users can define analysis arenas to restrict detection to biologically relevant regions.

### Automated output
IFA automatically exports:

- detection tables
- annotated images
- analysis session settings

---

# 🖥 Software Interface

| copy the path | terminal instruction |
|:-----------------------:|:----------------------:|
| ![](https://raw.githubusercontent.com/wenxiwang2000/Insect-Fecal-Analysis-IFA-/c9a0ba2d057e266a616f92406fb71e08739fc6f4/fithub2.png) | ![](https://github.com/wenxiwang2000/Insect-Fecal-Analysis-IFA-/blob/3bb2857d73cb77e5d6ed39efab8fc87974a98203/image.png) |

---

# ⚙️ Workflow

1. Drag images into the input panel.

2. Define a **region of interest (ROI)** by drawing a selection box.

3. Adjust **pixel threshold parameters** (maximum–minimum range) to refine detection.

4. Tune the **sensitivity parameter** to optimize detection of faint deposits.

5. Optionally apply an **HSV color mask** to exclude background pixels.

6. Export quantitative results and annotated images.

---

# 📊 Background Exclusion Example

HSV masking can be used to remove background artifacts and improve segmentation accuracy.

![](https://github.com/wenxiwang2000/Insect-Fecal-Analysis-IFA-/blob/adcd76f877462318c70e0fef772cdb686fa91017/git2.png)

---

# ▶ Running the Application

Ensure **Python** is installed on your system.


Then run:

```bash
streamlit run main.py



Original paper:
Wayland, M. T., Defaye, A., Rocha, J., Jayaram, S. A., Royet, J., Miguel-Aliaga, I., 
    Leulier, F., & Cognigni, P. (2014). Spotting the differences: Probing host/microbiota 
    interactions with a dedicated software tool for the analysis of faecal outputs in 
    Drosophila. Journal of Insect Physiology, 69, 126–135. 
    https://doi.org/10.1016/j.jinsphys.2014.05.023
