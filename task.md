Problem Statement:
In mineral exploration and ore processing, a rapid and objective assessment of raw material processing characteristics is critical. The traditional approach to ore classification based on panoramic photomicrographs of polished sections relies on a geologist's visual assessment, which presents several limitations:
- Subjectivity: different specialists may interpret sulfide intergrowth patterns and talc content differently, reducing inter-laboratory reproducibility.
- Labor intensity: manual segmentation and phase area calculation on high-resolution images (up to several gigapixels) take hours per sample.
- Scalability: when processing batches of dozens or hundreds of sections, manual analysis becomes a bottleneck in the research workflow.
- Difficulty of quantification: visual assessment of the "predominance" of standard versus fine intergrowths fails to provide the precise percentage values ​​needed to generate processing maps.
- Data variability: images vary in lighting, contrast, and polishing artifacts, complicating the use of rigid threshold-based algorithms.
Proposed Solution

Develop an end-to-end system for the automated classification of ores based on panoramic optical microscopy (OM) images of polished sections, capable of:
1. Segmenting and classifying sulfide inclusions:
Standard intergrowths — large, isolated sulfides with minimal replacement by a gray/dark phase (e.g., magnetite) → indicator of standard-grade ore or normal ore;
Fine intergrowths — sulfides significantly replaced by a non-sulfide phase → indicator of refractory (difficult-to-process) ore or hard ore.
2. Detecting and quantifying talc content — a dark, dispersed phase within the non-sulfide matrix, marked with a colored line in the training data. 
3. Applies expert classification logic:
	- If talc content > 10% → talc-bearing ore;
	- If talc content ≤ 10%:
⚬ Coarse intergrowths predominate → normal ore;
⚬ Fine intergrowths predominate → hard ore.
4. Generates an interpretable result:
Color overlay on the original image (green = coarse intergrowths, red = fine intergrowths, blue = talc);
Table with quantitative metrics: total sulfide content, proportion by intergrowth type, talc content;
Text output: "Ore classified as talc-bearing: talc content — 14%, prevalence of fine intergrowths — 62%."

Key Requirements
- Domain accuracy: The solution must correctly reflect the geological logic of the classification rather than simply maximizing segmentation metrics.
- Robustness to data variations: Ability to process images with varying lighting, contrast, and grinding/polishing artifacts.
- Interpretability: Geologists must be able to visually verify which areas are classified as talc or specific types of intergrowths.
- Practical integration: The solution must integrate into the existing laboratory workflow—from TIFF/PNG input to report export.
- Adaptability: Support for fine-tuning on new ore types or data from different microscopy equipment (transfer learning).

Functional Requirements
Image processing:
- Support for high-resolution formats: TIFF, PNG, JPEG;
- Automatic preprocessing: lighting normalization, noise reduction, contrast correction, and scaling for panoramic images;
- Pixel-level segmentation preserving inclusion morphology.
- Classification and quantitative analysis:
- Identification of sulfide phases (bright areas) against a silicate/oxide matrix (dark/gray areas);
- Classification of intergrowths based on the degree of replacement by the gangue phase;
- Detection of talc as a dispersed dark phase within the gangue matrix;
- Calculation of areas and percentages accounting for image scale.
Visualization and export:
- Color mask overlay on the original image (interactive view with zoom);
- Interface-based metrics table with CSV export capability;
- Generation of a brief text summary and PDF report export;
- Optional: Web interface using Streamlit/Gradio for user-friendly interaction. Batch processing and logging:
- Processing of image series without user intervention;
- Logging of analysis parameters to ensure reproducibility.

Non-functional requirements:
- Performance: processing of a single panoramic image (up to 10,000×10,000 pixels) — no more than 5 minutes on a workstation equipped with CPU/GPU.
- Reliability: correct processing of "challenging" cases — images with uneven lighting, scratches, or surface contamination on the thin section.
Accuracy:
- Talc fraction estimation error — no more than ±3% relative to expert annotation;
- Intergrowth type classification accuracy — F1-score of at least 90%.
Interface: intuitive for geologists without deep ML expertise; capability for manual mask correction (optional, for active learning mode).
Security: support for local deployment to handle confidential geological data.

Additional preferences
"Expert review" mode: capability for geologists to mark misclassified areas and add them to the retraining dataset.
Visual cues: display of not only the final mask but also the model's "confidence map" (probability heatmap) for ambiguous areas.
Documentation: detailed user manual with examples analyzing typical and borderline classification cases.
