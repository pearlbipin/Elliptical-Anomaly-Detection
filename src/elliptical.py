import numpy as np
from scipy.spatial.distance import cdist

class EllipticalAnomalyDetector:
    """
    Anomaly Detection Algorithm for Elliptical Clusters.
    Based on 'Anomaly Detection Algorithm for Elliptical Clusters Based On Maximum Cluster Diameter Criteria'
    by Pearl Bipin Pulickal and Ravi Prasad K.J.
    """

    def __init__(self, mode='simplified'):
        """
        Args:
            mode (str): 'simplified' for axis-aligned bounding box logic (Section 3.6).
                        'rotated' for the general case using max diameter (Section 3.5).
        """
        self.mode = mode
        self.center = None
        self.extreme_points = {}  # Stores PL, PR, PB, PT
        self.diameters = {}       # Stores DX (Major), DY (Minor)

    def fit(self, X):
        """
        Fit the model to the normal data points X to find cluster boundaries.
        """
        X = np.array(X)
        self.center = np.mean(X, axis=0)

        if self.mode == 'simplified':
            self._fit_simplified(X)
        elif self.mode == 'rotated':
            self._fit_rotated(X)
        else:
            raise ValueError("Mode must be 'simplified' or 'rotated'")

    def _fit_simplified(self, X):
        """
        Implementation of Section 3.6: Simplified Algorithm.
        Identifies boundaries using min/max coordinates (Axis-aligned).
        """
        # 1. Identify Endpoints of Major and Minor Axes [cite: 316-329]
        min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
        min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])

        # Store extreme boundaries
        self.extreme_points['min_x'] = min_x
        self.extreme_points['max_x'] = max_x
        self.extreme_points['min_y'] = min_y
        self.extreme_points['max_y'] = max_y

        # 3. Compare Differences to Axis Lengths [cite: 336-338]
        self.diameters['DX'] = max_x - min_x
        self.diameters['DY'] = max_y - min_y

    def _fit_rotated(self, X):
        """
        Implementation of Section 3.5: Rotated Algorithm.
        Identifies boundaries using Maximum Euclidean Distance (General Case).
        """
        # Step 1: Maximizing Euclidean Distance to find Major Axis 
        # Find the two points with the maximum squared Euclidean distance
        dists = cdist(X, X, metric='sqeuclidean')
        i, j = np.unravel_index(dists.argmax(), dists.shape)
        p1, p2 = X[i], X[j] # These are PL and PR

        # Store Major Axis endpoints and length
        self.extreme_points['PL'] = p1
        self.extreme_points['PR'] = p2
        self.diameters['DX'] = np.sqrt(dists[i, j])

        # Step 3: Determine Perpendicular Bisector to find Minor Axis [cite: 254-265]
        # Vector along major axis
        major_vec = p2 - p1
        if np.linalg.norm(major_vec) == 0:
            major_vec_norm = np.array([1, 0])
        else:
            major_vec_norm = major_vec / np.linalg.norm(major_vec)

        # Orthogonal vector (-y, x) for 2D plane
        minor_vec_norm = np.array([-major_vec_norm[1], major_vec_norm[0]])

        # Project all points onto the minor vector relative to center to find max spread
        vecs_from_center = X - self.center
        projections = np.dot(vecs_from_center, minor_vec_norm)
        
        # Identify points with min and max projection (bottom-most and top-most relative to rotation)
        idx_min = np.argmin(projections)
        idx_max = np.argmax(projections)
        
        # Store Minor Axis endpoints and length
        p3 = X[idx_min]
        p4 = X[idx_max]
        
        self.extreme_points['PB'] = p3
        self.extreme_points['PT'] = p4
        self.diameters['DY'] = np.linalg.norm(p4 - p3)

    def predict(self, X):
        """
        Predict if points in X are anomalies.
        Returns: 1 for Anomaly, 0 for Normal.
        """
        X = np.array(X)
        predictions = []

        for point in X:
            is_anomaly = False

            if self.mode == 'simplified':
                # Simplified Logic [cite: 340-344]
                # Calculate absolute coordinate differences
                dl = np.abs(point[0] - self.extreme_points['min_x'])
                dr = np.abs(point[0] - self.extreme_points['max_x'])
                db = np.abs(point[1] - self.extreme_points['min_y'])
                dt = np.abs(point[1] - self.extreme_points['max_y'])
                
                dx = self.diameters['DX']
                dy = self.diameters['DY']

                # Pearl's Heuristic (Simplified): Anomaly if difference > axis length
                if (dl > dx) or (dr > dx) or (db > dy) or (dt > dy):
                    is_anomaly = True

            elif self.mode == 'rotated':
                # Rotated Logic 
                # Calculate Euclidean distances to the identified extreme points
                dl = np.linalg.norm(point - self.extreme_points['PL'])
                dr = np.linalg.norm(point - self.extreme_points['PR'])
                db = np.linalg.norm(point - self.extreme_points['PB'])
                dt = np.linalg.norm(point - self.extreme_points['PT'])

                dx = self.diameters['DX']
                dy = self.diameters['DY']

                # Pearl's Heuristic (Rotated): Anomaly if distance to endpoint > axis diameter
                if (dl > dx) or (dr > dx) or (db > dy) or (dt > dy):
                    is_anomaly = True

            predictions.append(1 if is_anomaly else 0)

        return np.array(predictions)
