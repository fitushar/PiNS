# PiNS (Point-driven Nodule Segmentation) - Technical Documentation
## Professional Technical Documentation

### Version: 1.0.0
### Authors: Fakrul Islam Tushar (tushar.ece@dule.edu)
### Date: September 2025
### License: CC-BY-NC-4.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Technical Specifications](#technical-specifications)
5. [API Reference](#api-reference)
6. [Implementation Details](#implementation-details)
7. [Performance Metrics](#performance-metrics)
8. [Clinical Applications](#clinical-applications)
9. [Validation](#validation)
10. [Future Developments](#future-developments)

---

## Overview

### Abstract

PiNS (Point-driven Nodule Segmentation) is a  medical imaging toolkit designed for automated detection, segmentation, and analysis of pulmonary nodules in computed tomography (CT) scans. The toolkit provides an end-to-end pipeline from coordinate-based nodule identification to quantitative radiomics feature extraction.

### Key Capabilities

**1. Automated Nodule Segmentation**
- K-means clustering-based segmentation with configurable expansion
- Multi-algorithm support (K-means, Gaussian Mixture Models, Fuzzy C-Means, Otsu)
- Sub-voxel precision coordinate handling
- Adaptive region growing with millimeter-based expansion

**2. Quantitative Radiomics Analysis**
- PyRadiomics-compliant feature extraction
- 100+ standardized imaging biomarkers
- IBSI-compatible feature calculations
- Configurable intensity normalization and resampling

**3. Patch-based Data Preparation**
- 3D volumetric patch extraction (64³ default)
- Standardized intensity windowing for lung imaging
- Deep learning-ready data formatting
- Automated coordinate-to-voxel transformation

### Clinical Significance

PiNS addresses critical challenges in pulmonary nodule analysis:
- **Reproducibility**: Standardized segmentation protocols
- **Quantification**: Objective radiomics-based characterization
- **Scalability**: Batch processing capabilities for research cohorts
- **Interoperability**: NIfTI support with Docker containerization

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    PiNS Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                                │
│  ├── CT DICOM/NIfTI Images                                  │
│  ├── Coordinate Annotations (World/Voxel)                   │
│  └── Configuration Parameters                               │
├─────────────────────────────────────────────────────────────┤
│  Processing Layer                                           │
│  ├── Image Preprocessing                                    │
│  │   ├── Intensity Normalization                           │
│  │   ├── Resampling & Interpolation                        │
│  │   └── Coordinate Transformation                         │
│  ├── Segmentation Engine                                    │
│  │   ├── K-means Clustering                                │
│  │   ├── Region Growing                                     │
│  │   └── Morphological Operations                          │
│  └── Feature Extraction                                     │
│      ├── Shape Features                                     │
│      ├── First-order Statistics                            │
│      ├── Texture Features (GLCM, GLRLM, GLSZM, GLDM)       │
│      └── Wavelet Features                                   │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                               │
│  ├── Segmentation Masks (NIfTI)                            │
│  ├── Quantitative Features (CSV)                           │
│  ├── Image Patches (NIfTI)                                 │
│  └── Processing Logs                                        │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Containerization**: Docker (Ubuntu 20.04 base)
- **Medical Imaging**: SimpleITK, PyRadiomics 3.1.0+
- **Scientific Computing**: NumPy, SciPy, scikit-learn
- **Data Management**: Pandas, NiBabel
- **Visualization**: Matplotlib
- **Languages**: Python 3.8+, Bash scripting

---

## Core Components

### Component 1: Nodule Segmentation Pipeline

**Script**: `DLCS24_KNN_2mm_Extend_Seg.sh`
**Purpose**: Automated segmentation of pulmonary nodules from coordinate annotations

**Algorithm Workflow**:
1. **Coordinate Processing**: Transform world coordinates to voxel indices
2. **Region Initialization**: Create bounding box around nodule center
3. **Clustering Segmentation**: Apply K-means with k=2 (nodule vs. background)
4. **Connected Component Analysis**: Extract largest connected component
5. **Morphological Refinement**: Apply expansion based on clinical parameters
6. **Quality Control**: Validate segmentation size and connectivity

**Technical Parameters**:
- Expansion radius: 2.0mm (configurable)
- Clustering algorithm: K-means (alternatives: GMM, FCM, Otsu)
- Output format: NIfTI (.nii.gz)
- Coordinate system: ITK/SimpleITK standard

### Component 2: Radiomics Feature Extraction

**Script**: `DLCS24_KNN_2mm_Extend_Radiomics.sh`
**Purpose**: Quantitative imaging biomarker extraction from segmented nodules

**Feature Categories**:
1. **Shape Features (14 features)**
   - Sphericity, Compactness, Surface Area
   - Volume, Maximum Diameter
   - Elongation, Flatness

2. **First-order Statistics (18 features)**
   - Mean, Median, Standard Deviation
   - Skewness, Kurtosis, Entropy
   - Percentiles (10th, 90th)

3. **Second-order Texture (75+ features)**
   - Gray Level Co-occurrence Matrix (GLCM)
   - Gray Level Run Length Matrix (GLRLM)
   - Gray Level Size Zone Matrix (GLSZM)
   - Gray Level Dependence Matrix (GLDM)

4. **Higher-order Features (100+ features)**
   - Wavelet decomposition features
   - Laplacian of Gaussian filters

**Normalization Protocol**:
- Bin width: 25 HU
- Resampling: 1×1×1 mm³
- Interpolation: B-spline (image), Nearest neighbor (mask)

### Component 3: Patch Extraction Pipeline

**Script**: `DLCS24_CADe_64Qpatch.sh`
**Purpose**: 3D volumetric patch extraction for deep learning applications

**Patch Specifications**:
- **Dimensions**: 64×64×64 voxels (configurable)
- **Centering**: World coordinate-based positioning
- **Windowing**: -1000 to 500 HU (lung window)
- **Normalization**: Min-max scaling to [0,1]
- **Boundary Handling**: Zero-padding for edge cases

**Output Format**:
- Individual NIfTI files per nodule
- CSV metadata with coordinates and labels
- Standardized naming convention

---

## Technical Specifications

### Hardware Requirements

**Minimum Requirements**:
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Storage: 50 GB available space
- Docker: 20.10.0+

**Recommended Configuration**:
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16+ GB
- Storage: 100+ GB SSD
- GPU: CUDA-compatible (for future ML extensions)

### Input Data Requirements

**Image Specifications**:
- Format: NIfTI
- Modality: CT (chest)
- Resolution: 0.5-2.0 mm³ voxel spacing
- Matrix size: 512×512 or larger
- Bit depth: 16-bit signed integers

**Annotation Format**:
```csv
ct_nifti_file,nodule_id,coordX,coordY,coordZ,w,h,d,Malignant_lbl
DLCS_0001.nii.gz,DLCS_0001_01,-106.55,-63.84,-211.68,4.39,4.39,4.30,0
```

**Required Columns**:
- `ct_nifti_file`: Image filename
- `coordX/Y/Z`: World coordinates (mm)
- `w/h/d`: Bounding box dimensions (mm) 
- `Malignant_lbl`: Binary label (optional)





## API Reference

### Bash Script Interface

#### Segmentation Script
```bash
./scripts/DLCS24_KNN_2mm_Extend_Seg.sh
```

**Configuration Variables**:
```bash
DATASET_NAME="DLCSD24"              # Dataset identifier
SEG_ALG="knn"                       # Segmentation algorithm
EXPANSION_MM=2.0                    # Expansion radius (mm)
RAW_DATA_PATH="/app/demofolder/data/DLCS24/"
DATASET_CSV="/app/demofolder/data/DLCSD24_Annotations_N2.csv"
```

#### Radiomics Script
```bash
./scripts/DLCS24_KNN_2mm_Extend_Radiomics.sh
```

**Additional Parameters**:
```bash
EXTRACT_RADIOMICS_FLAG="--extract_radiomics"
PARAMS_JSON="/app/scr/Pyradiomics_feature_extarctor_pram.json"
```

#### Patch Extraction Script
```bash
./scripts/DLCS24_CADe_64Qpatch.sh
```

**Patch Parameters**:
```bash
PATCH_SIZE="64 64 64"               # Voxel dimensions
NORMALIZATION="-1000 500 0 1"       # HU window and output range
CLIP="True"                         # Enable intensity clipping
```

### Python API (Internal)

#### Segmentation Function
```python
def candidateSeg_main():
    """
    Main segmentation pipeline
    
    Parameters:
    -----------
    raw_data_path : str
        Path to input CT images
    dataset_csv : str  
        Path to coordinate annotations
    seg_alg : str
        Segmentation algorithm {'knn', 'gmm', 'fcm', 'otsu'}
    expansion_mm : float
        Expansion radius in millimeters
        
    Returns:
    --------
    None (saves masks to disk)
    """
```

#### Radiomics Function
```python
def seg_pyradiomics_main():
    """
    Radiomics feature extraction pipeline
    
    Parameters:
    -----------
    params_json : str
        PyRadiomics configuration file
    extract_radiomics : bool
        Enable feature extraction
        
    Returns:
    --------
    features : DataFrame
        Quantitative imaging features
    """
```

---

## Implementation Details

### Docker Container Specifications

**Base Image**: `ft42/pins:latest`
**Size**: ~11 GB (includes CUDA libraries)
**Dependencies**:
```dockerfile
# Core medical imaging libraries
SimpleITK==2.4+
pyradiomics==3.1.0
scikit-learn==1.3.0

# Deep learning and computer vision
torch==2.8.0
torchvision==0.23.0
monai==1.4.0
opencv-python-headless==4.11.0

# Scientific computing
numpy==1.24.4
scipy
pandas
scipy==1.11.1
nibabel==5.1.0

# Data processing
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.1

# Utilities
tqdm==4.65.0
```

### File Organization

```
PiNS/
├── scripts/
│   ├── DLCS24_KNN_2mm_Extend_Seg.sh
│   ├── DLCS24_KNN_2mm_Extend_Radiomics.sh
│   └── DLCS24_CADe_64Qpatch.sh
├── scr/
│   ├── candidateSeg_pipiline.py
│   ├── candidateSeg_radiomicsExtractor_pipiline.py
│   ├── candidate_worldCoord_patchExtarctor_pipeline.py
│   ├── cvseg_utils.py
│   └── Pyradiomics_feature_extarctor_pram.json
├── demofolder/
│   ├── data/
│   │   ├── DLCS24/
│   │   └── DLCSD24_Annotations_N2.csv
│   └── output/
└── docs/
    ├── README.md
    ├── TECHNICAL_DOCS.md
    └── HUGGINGFACE_CARD.md
```

### Configuration Management

**PyRadiomics Parameters** (`Pyradiomics_feature_extarctor_pram.json`):
```json
{
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "interpolator": "sitkBSpline",
    "labelInterpolator": "sitkNearestNeighbor"
}
```

**Segmentation Parameters**:
- K-means clusters: 2 (nodule vs background)
- Connected component threshold: Largest component
- Morphological operations: Binary closing with 1mm kernel

---



### Computational Efficiency

**Processing Time Analysis**:
- Segmentation: 15-30 seconds per nodule
- Radiomics extraction: 5-10 seconds per mask
- Patch extraction: 2-5 seconds per patch
- Total pipeline: <2 minutes per case

**Scalability Analysis**:
- Linear scaling with nodule count
- Memory usage: ~500 MB per concurrent image
- Disk I/O: ~50 MB/s sustained throughput
- CPU utilization: 85-95% (multi-threaded operations)

---

## Research Applications

### Diagnostic Imaging

**Lung Cancer Screening**:
- Automated nodule characterization
- Growth assessment in follow-up studies
- Risk stratification based on radiomics profiles

**Research Applications**:
- Biomarker discovery studies
- Machine learning dataset preparation
- Multi-institutional validation studies

### Integration Pathways

**AI Pipeline Integration**:
- Preprocessed patch data for CNNs
- Feature vectors for traditional ML
- Standardized evaluation protocols

---



## License and Usage Terms

### Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0)

**Permitted Uses**:
- Research and educational purposes
- Academic publications and presentations
- Non-commercial clinical research
- Open-source contributions and modifications

**Requirements**:
- Attribution to original authors and PiNS toolkit
- Citation of relevant publications
- Sharing of derivative works under same license
- Clear indication of any modifications made

**Restrictions**:
- Commercial use requires separate licensing agreement
- No warranty or liability provided
- Contact ft42@research.org for commercial licensing

**Citation Requirements**:

```bibtex
@software{pins2025,
  title={PiNS: Point-driven Nodule Segmentation Toolkit },
  author={Fakrul Islam Tushar},
  year={2025},
  url={https://github.com/fitushar/PiNS},
  version={1.0.0},
  doi={10.5281/zenodo.17171571},
  license={CC-BY-NC-4.0}
}
```

---

## Validation & Quality Assurance

**Evaluation Criteria:** In the absence of voxel-level ground truth, we adopted a bounding box–supervised evaluation strategy to assess segmentation performance. Each CT volume was accompanied by annotations specifying the nodule center in world coordinates and its dimensions in millimeters, which were converted into voxel indices using the image spacing and clipped to the volume boundaries. A binary mask representing the bounding box was then constructed and used as a weak surrogate for ground truth. we extracted a patch centered on the bounding box, extending it by a fixed margin (64 voxels) to define the volume of interest (VOI). Predicted segmentation masks were cropped to the same VOI-constrained region of interest, and performance was quantified in terms of Dice similarity coefficient. Metrics were computed per lesion. This evaluation strategy enables consistent comparison of segmentation algorithms under weak supervision while acknowledging the limitations of not having voxel-level annotations.

Segmentation performance of **KNN (ours PiNS)**, **VISTA3D auto**, and **VISTA3D points** ([He et al. 2024](https://github.com/Project-MONAI/VISTA/tree/main/vista3d)) across different nodule size buckets. (top) Bar plots display the mean Dice similarity coefficient for each model and size category. (buttom) Boxplots show the distribution of Dice scores, with boxes representing the interquartile range, horizontal lines indicating the median, whiskers extending to 1.5× the interquartile range, and circles denoting outliers.

<p align="center">
  <img src="assets/Segmentation_Evaluation_KNNVista3Dauto_DLCS24_HIST.png" alt="(a)" width="700">
</p>

<p align="center">
  <img src="assets/Segmentation_Evaluation_KNNVista3Dauto_DLCS24_BOX.png" alt="(b)" width="700">
</p>
## Limitations & Considerations

### Current Limitations
- **Nodule Size**: Optimized for nodules 3-30mm diameter
- **Image Quality**: Requires standard clinical CT protocols
- **Coordinate Accuracy**: Dependent on annotation precision
- **Processing Time**: Sequential processing (parallelization possible)

## Contributing & Development

### Research Collaborations
We welcome collaborations from:
- **Academic Medical Centers**
- **Radiology Departments** 
- **Medical AI Companies**
- **Open Source Contributors**





### Related Publications
1. **AI in Lung Health: Benchmarking** : [Tushar et al. arxiv (2024)](https://arxiv.org/abs/2405.04605)
2. **AI in Lung Health: Benchmarking** : [https://github.com/fitushar/AI-in-Lung-Health-Benchmarking](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets)
4. **DLCS Dataset**: [Wang et al. Radiology AI 2024](https://doi.org/10.1148/ryai.240248);[Zenedo](https://zenodo.org/records/13799069)
5. **SYN-LUNGS**: [Tushar et al., arxiv 2025](https://arxiv.org/abs/2502.21187)
6. **Refining Focus in AI for Lung Cancer:** Comparing Lesion-Centric and Chest-Region Models with Performance Insights from Internal and External Validation. [![arXiv](https://img.shields.io/badge/arXiv-2411.16823-<color>.svg)](https://arxiv.org/abs/2411.16823)
7. **Peritumoral Expansion Radiomics** for Improved Lung Cancer Classification. [![arXiv](https://img.shields.io/badge/arXiv-2411.16008-<color>.svg)](https://arxiv.org/abs/2411.16008)
8. **PyRadiomics Framework**: [van Griethuysen et al., Cancer Research 2017](https://pubmed.ncbi.nlm.nih.gov/29092951/)



## License & Usage
**license: cc-by-nc-4.0**
### Academic Use License
This project is released for **academic and non-commercial research purposes only**.  
You are free to use, modify, and distribute this code under the following conditions:
- ✅ Academic research use permitted
- ✅ Modification and redistribution permitted for research
- ❌ Commercial use prohibited without prior written permission
For commercial licensing inquiries, please contact: tushar.ece@duke.edu


## Support & Community

### Getting Help
- **📖 Documentation**: [Comprehensive technical docs](https://github.com/fitushar/PiNS/blob/main/docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/fitushar/PiNS/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/fitushar/PiNS/discussions)
- **📧 Email**: tushar.ece@Duke.edu ; fitushar.mi@gmail.com

### Community Stats
- **Publications**: 5+ research papers
- **Contributors**: Active open-source community

