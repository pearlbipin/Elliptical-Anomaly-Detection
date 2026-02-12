# Anomaly Detection Algorithm for Elliptical Clusters Based On Maximum Cluster Diameter Criteria

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

## 📜 Abstract

This repository contains the reference implementation of the research paper **"Anomaly Detection Algorithm for Elliptical Clusters Based On Maximum Cluster Diameter Criteria"**.

The algorithm introduces a deterministic geometric heuristic—**Pearl's Heuristic**—to detect anomalies within elliptical clusters. Unlike probabilistic or density-based methods (like LOF or DBSCAN), this approach establishes definitive geometric criteria for identifying anomalies, achieving high performance in scenarios with rotated or axis-aligned elliptical distributions.

## 🚀 Key Features

* **Pearl's Heuristic:** A novel geometric rule stating that a point lies outside an ellipse if its distance to the cluster's extreme axial points exceeds the major/minor axis lengths.
* **Two Operational Modes:**
    * **Simplified Algorithm:** An $O(N)$ approach for axis-aligned clusters using bounding box logic.
    * **Rotated Algorithm:** A generalized approach that dynamically identifies the major axis via maximum Euclidean distance and the minor axis via perpendicular bisectors.
* **High Performance:** Achieved 100% Precision, Recall, and F1 Scores on synthetic test datasets, outperforming DBSCAN and Isolation Forest in specific elliptical scenarios.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Elliptical-Anomaly-Detection.git](https://github.com/YourUsername/Elliptical-Anomaly-Detection.git)

# Navigate to the directory
cd Elliptical-Anomaly-Detection

# Install dependencies
pip install -r requirements.txt

```

## 🧠 Mathematical Foundation

The core of the algorithm relies on **Pearl's Heuristic**.

Given a point  and an ellipse  with major axis length  and minor axis length :

* Let  be the distances from  to the extreme points of the ellipse (Left, Right, Bottom, Top).
*  is considered an anomaly if **at least one** of the following holds:

For **Rotated Clusters**, the algorithm calculates  by maximizing the Euclidean distance between all pairs of points  and  in the cluster.

## 💻 Usage

### 1. Simplified Mode (Axis-Aligned)

Best for quick approximations where clusters are aligned with  axes.

```python
import numpy as np
from src.elliptical import EllipticalAnomalyDetector

# Generate synthetic data
X_train = np.random.rand(100, 2) 

# Initialize and fit
model = EllipticalAnomalyDetector(mode='simplified')
model.fit(X_train)

# Detect anomalies
# Returns: 1 for Anomaly, 0 for Normal
predictions = model.predict(X_test)

```

### 2. Rotated Mode (General Case)

Best for precise detection in complex, rotated datasets.

```python
# Initialize with rotated mode
model = EllipticalAnomalyDetector(mode='rotated')
model.fit(X_train)

predictions = model.predict(X_test)

```

## 📊 Performance

In comparative simulations against standard industry algorithms on elliptical datasets, this model demonstrated superior metrics:

| Metric | Pearl's Algorithm | DBSCAN | Isolation Forest | LOF |
| --- | --- | --- | --- | --- |
| **Accuracy** | **1.00** | 0.77 | 0.84 | 0.97 |
| **Precision** | **1.00** | 0.07 | 0.14 | 0.56 |
| **Recall** | **1.00** | 0.40 | 0.63 | 0.90 |
| **F1 Score** | **1.00** | 0.12 | 0.23 | 0.69 |

## 🔗 Citation

If you use this code or methodology in your research, please cite the original paper:

> **Pulickal, P. B., & Prasad, R. K. J. (2024).** *Anomaly Detection Algorithm for Elliptical Clusters Based On Maximum Cluster Diameter Criteria.*

## 👥 Authors

* **Pearl Bipin Pulickal** - *Primary Contributor*
* **Ravi Prasad K.J.** - *Associate Professor of Mathematics, NIT Goa*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
