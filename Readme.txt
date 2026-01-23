README — TURD-based Spot Detector (Refined Sensitivity)

This software is a refined implementation based on T.U.R.D. (The Ultimate Reader of Dung), a method for detecting and quantifying insect fecal deposits using adaptive, local pixel-intensity comparisons.

In the original T.U.R.D. framework, detection sensitivity is implicitly fixed by predefined adaptive threshold parameters and cannot be directly adjusted by the user.
In this version, we introduce an explicit user-adjustable sensitivity parameter that controls the local contrast comparison between each pixel and its surrounding neighborhood.

By exposing sensitivity as a tunable parameter, this implementation allows more flexible and robust detection, enabling reliable identification of very small and faint fecal deposits across a wide range of insect species, while preserving the core detection logic and measurement principles of the original T.U.R.D. method.

This software is intended for research and exploratory analysis and should be cited with reference to the original T.U.R.D. publication.
