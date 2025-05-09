# Beyond the Bubble: AI That Stops Retinal Slip Before It Steals Your Sight

<p align="center">
  <img src="Retinal%20Detachment.png" alt="Retinal Detachment Illustration" width="800"/>
</p>

## Project Overview

This project addresses a critical challenge in ophthalmology: unintentional retinal displacement following rhegmatogenous retinal detachment (RRD) repair. While current surgical techniques achieve high anatomical success rates (up to 96%), studies show that up to 62.8% of cases experience retinal displacement, leading to metamorphopsia (distorted vision) and other visual impairments.

**Problem**: Existing methods for detecting retinal displacement—such as FAF-based retinal vessel imprinting and OCT-based homography—suffer from significant drawbacks:
- Detection only after tamponade agent resorption (delays of 2-12 weeks)
- Low recall in macular regions
- Dependency on manual landmarking
- Need for clear post-operative scans

<p align="center">
  <img src="Retinal%20Displacement.jpg" alt="Retinal Displacement Illustration by OCT-based homography" width="300"/>
</p>

**Solution**: The first fully automated AI system that can predict vasculature patterns and detect displacement in post-operative fundus images before the tamponade agent has settled, reducing the diagnostic window from weeks to seconds.

<p align="center">
  <img src="System%20Design.png" alt="System Design" width="600"/>
</p>

## The Clinical Impact

Retinal detachment affects approximately 12.7 per 100,000 people annually worldwide, with the incidence rising by 5.4 per 100,000 per decade. Patients with high myopia have a 10x higher risk, and up to 44% of RRD cases have had cataract surgery. While surgical repair techniques like pneumatic retinopexy (PR), scleral buckling (SB), and pars plana vitrectomy (PPV) have high anatomical success rates, displacement of the retina relative to its original position remains a significant clinical challenge.

As documented across multiple studies, 34-62.8% of patients experience retinal displacement that can cause:
- Metamorphopsia (straight lines appearing distorted)
- Double vision (diplopia)
- Reduced visual acuity
- Functional vision loss

By enabling real-time detection of displacement immediately post-operation, this solution allows surgeons to adjust patient positioning when it matters most, potentially preventing permanent vision impairment.

## Technical Approach

The system consists of three primary components:

### 1. Vessel Isolation

<p align="center">
  <img src="Vessel%20Isolation%20Design.png" alt="Vessel Isolation Design" width="350"/>
</p>

I developed a novel vessel isolation technique using LAB-weighted preprocessing with CLAHE to enhance vessel contrast, especially in macular regions. This method outperforms existing state-of-the-art approaches with:
- **94.6%** average accuracy
- **82.3%** sensitivity
- **94.8%** specificity
- **0.16s** runtime

The process includes:
- LAB color space weighting for enhanced contrast
- CLAHE for adaptive local contrast enhancement
- Background suppression via average filtering
- Inter-means thresholding with optimized bias
- Morphological operations for cleaning

### 2. Optic Disc Detection

<p align="center">
  <img src="Optic%20Disc%20Detection%20Design.png" alt="Optic Disc Detection Design" width="500"/>
</p>

A physics-inspired gravitational-edge detection algorithm locates the optic disc as an anatomical reference point, achieving:
- **95%** average accuracy on the STARE dataset
- **100%** accuracy on the HRF dataset
- **0.44s** average runtime

The algorithm incorporates:
- Adaptive gamma correction based on local contrast
- Anisotropic diffusion for edge preservation
- Gravitational-edge detection using intensity gradients
- Candidate selection with mean distance-based filtering

### 3. Feature Evaluation
Extraction of geometric graphical features from the vessel network provides the foundation for AI model training:
- Bifurcations and endpoints
- Branch lengths
- Turning angles
- Conversion to polar coordinates with the optic disc as origin

## Results & Validation

The system was thoroughly evaluated using three standard datasets:

### STARE Database
- **Accuracy**: 95.18%
- **Sensitivity**: 83.92%
- **Specificity**: 94.32%
- **Runtime**: 0.12s

### DRIVE Database
- **Accuracy**: 95.70%
- **Sensitivity**: 75.89%
- **Specificity**: 95.79%
- **Runtime**: 0.16s

### HRF Database
- **Accuracy**: 93.12%
- **Sensitivity**: 86.13%
- **Specificity**: 94.34%
- **Runtime**: 0.22s

The proposed method demonstrates significant improvements over existing techniques, particularly in processing speed and detection sensitivity in macular regions.

## Future Development

This project lays the groundwork for several exciting applications:

1. **AI Model Completion** - Implementing a Graph Neural Network (GNN) to predict the vessel pattern and quantify displacement direction and magnitude.

2. **Synthetic Data Generation** - Using extracted features to generate realistic synthetic vessels via turning angle-based path reconstruction and mapping line functions.

3. **Real-Time Clinical Guidance** - Intraoperative AI guidance to help surgeons position the retina correctly during surgery.

4. **Home Monitoring System** - Integration with portable fundus cameras to allow patients to check their eyes at home post-surgery, with AI recommendations for optimal sleeping positions.

5. **3D Visualization** - Further development of a 3D model prototype that simulates light propagation through the eye, helping visualize displacement in three-dimensional space.

<p align="center">
  <img src="3D%20Overlay%20Prototype.png" alt="3D Overlay Prototype" width="450"/>
</p>

## Repository Structure

```
├── Vessel_Isolation.py        # Core implementation for vessel segmentation
├── Optic_Disc_Detection.py    # Gravitational edge detection for optic disc
├── Feature_Evaluation.py      # Extract bifurcations, endpoints, and angles
├── Performance_Metrics/
│   ├── STARE_Performance.md   # Results on STARE database for Vessel Isolation
│   ├── DRIVE_Performance.md   # Results on DRIVE database for Vessel Isolation
│   ├── HRF_Performance.md   # Results on DRIVE database for Vessel Isolation
│   ├── STARE OD Detection Performance.md   # Results on STARE database for Optic Disc Detection
│   └── HRF OD Detection Performance.md     # Results on HRF database for Optic Disc Detection
└── 3D_Prototype/              # Preliminary 3D visualization code
```
## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Shreyas8612/EEE3094.git
cd EEE3094

# Install dependencies
pip install -r requirements.txt

# Run vessel isolation on a sample image
python Vessel_Isolation.py --image path/to/fundus/image.jpg

# Run optic disc detection
python Optic_Disc_Detection.py --image path/to/fundus/image.jpg
```

## Publications & References

This work builds upon and extends research from multiple domains:

1. Xiong et al. (2025) - Rhegmatogenous retinal detachment review
2. Ge et al. (2023) - International incidence of RRD
3. Shiragami et al. (2009) - Unintentional displacement after vitrectomy
4. Brosh et al. (2024) - OCT-based homography for displacement detection
5. Alshayeji et al. (2016) - Gravitational edge detection
6. Sule & Viriri (2022) - Contrast enhancement for vessel segmentation

## Acknowledgments

First and foremost, I would like to express my deepest gratitude to my supervisor, Professor Patrick Degenaar, for his invaluable guidance and continuous support throughout this project. I would also like to thank Dr. Tafadzwa Young-Zvandasara for his clinical perspective that helped shape the application of this work.

I'm grateful to the creators of the STARE, DRIVE, and HRF datasets for providing the resources necessary for validation.

---

*This project was completed as part of a dissertation for the degree of Electrical and Electronic Engineering at Newcastle University, May 2025.*