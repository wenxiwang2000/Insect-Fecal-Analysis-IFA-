README — TURD-based Spot Detector (Refined Sensitivity)

This software is a refined implementation based on T.U.R.D. (The Ultimate Reader of Dung), a method for detecting and quantifying insect fecal deposits using adaptive, local pixel-intensity comparisons.

In the original T.U.R.D. framework, detection sensitivity is implicitly fixed by predefined adaptive threshold parameters and cannot be directly adjusted by the user.
In this version, we introduce an explicit user-adjustable sensitivity parameter that controls the local contrast comparison between each pixel and its surrounding neighborhood.

By exposing sensitivity as a tunable parameter, this implementation allows more flexible and robust detection, enabling reliable identification of very small and faint fecal deposits across a wide range of insect species, while preserving the core detection logic and measurement principles of the original T.U.R.D. method.

This software is intended for research and exploratory analysis and should be cited with reference to the original T.U.R.D. publication.

Steps: copy the path of this main.py

![image alt](https://raw.githubusercontent.com/wenxiwang2000/Insect-Fecal-Analysis-IFA-/c9a0ba2d057e266a616f92406fb71e08739fc6f4/fithub2.png)

remember downlaod python in your computer, and run as streamlit run main.py
![image alt](https://github.com/wenxiwang2000/Insect-Fecal-Analysis-IFA-/blob/3bb2857d73cb77e5d6ed39efab8fc87974a98203/image.png)

Drag images into the input field. Begin adjusting pixel arrangement (maximum-minimum values) by drawing a selection area on the image panel. Combine this with sensitivity parameters to select optimal detection conditions. Our AI-powered background removal technology ensures detection accuracy remains unaffected by the original image's background color!
![image alt](https://github.com/wenxiwang2000/Insect-Fecal-Analysis-IFA-/blob/adcd76f877462318c70e0fef772cdb686fa91017/git2.png)



software is made by ourself but the parameter is based on T.U.R.D paper:
Wayland, M. T., Defaye, A., Rocha, J., Arcot Jayaram, S., Royet, J., Miguel-Aliaga, I., Leulier, F., & Cognigni, P. (2014). Spotting the differences: Probing host/microbiota interactions with a dedicated software tool for the analysis of faecal outputs in Drosophila. Journal of Insect Physiology, 69, 126–135. https://doi.org/10.1016/j.jinsphys.2014.05.023
