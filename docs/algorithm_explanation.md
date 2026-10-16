# Detailed Mathematical and Algorithmic Explanation

This document provides in-depth mathematical formulations, OpenCV implementation specifics, and parameter derivations for the digital image processing algorithms utilized in the **Smart Coin Detection System**.

---

## 1. Grayscale Luminance Conversion
### Mathematical Formulation
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
### Implementation Details
- **OpenCV Function:** `cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)`
- **Rationale:** Optical cameras capture RGB color spaces where red, green, and blue sensors respond with different sensitivities. Because the human eye's cones are most sensitive to green wavelengths ($550$ nm), standard luminance $Y$ weights green by $58.7\%$. Grayscale conversion reduces data bandwidth from $3$ channels to $1$ intensity channel, accelerating edge calculation by $300\%$.

---

## 2. 2D Gaussian Filtering (Denoising)
### Mathematical Formulation
$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
The discrete $k \times k$ convolution matrix $K$ is formed by sampling $G(x, y)$ at discrete grid integer offsets and normalizing such that $\sum K_{i,j} = 1$.
### Implementation Details
- **OpenCV Function:** `cv2.GaussianBlur(gray, (9, 9), sigmaX=1.5, sigmaY=1.5)`
- **Rationale:** High-frequency camera noise produces steep false edge gradients. A Gaussian filter provides isotropic smoothing, meaning spatial smoothing is equal in all directions. A standard deviation of $\sigma=1.5$ effectively removes surface grain without displacing edge boundaries.

---

## 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
### Algorithmic Steps
1. The grayscale image is partitioned into $M \times N$ non-overlapping contextual tiles ($8 \times 8$).
2. A localized histogram $h_k(i)$ is computed for each tile $k$.
3. Any histogram bin exceeding clip limit $\beta = 2.0$ has its excess pixels truncated and redistributed uniformly:
   $$N_{\text{clipped}} = \sum \max(0, h_k(i) - \beta)$$
   $$h_k^{\text{equalized}}(i) = \min(h_k(i), \beta) + \frac{N_{\text{clipped}}}{\text{Total Bins}}$$
4. Cumulative distribution functions (CDF) are constructed.
5. Bilinear interpolation between neighboring tile mappings prevents boundary artifacts.
### Implementation Details
- **OpenCV Function:** `clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); enhanced = clahe.apply(image)`
- **Rationale:** Global histogram equalization causes washed-out backgrounds and extreme glare amplification. CLAHE boosts local contrast at coin rims while keeping background noise strictly bounded.

---

## 4. Canny Edge Detection & Hysteresis Tracking
### Algorithmic Pipeline
1. **Sobel Convolution:** Computes horizontal ($G_x$) and vertical ($G_y$) gradients:
   $$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$
2. **Non-Maximum Suppression (NMS):** Thins edge ridges by preserving only gradient values that are local maxima along the normal direction $\theta$.
3. **Hysteresis Thresholding:** Uses dual thresholds ($T_{\text{low}} = 40, T_{\text{high}} = 120$):
   - Pixel with $|G| \ge T_{\text{high}}$: Strong edge (retained).
   - Pixel with $|G| < T_{\text{low}}$: Suppressed (discarded).
   - Pixel with $T_{\text{low}} \le |G| < T_{\text{high}}$: Weak edge (retained only if connected to a strong edge).
### Implementation Details
- **OpenCV Function:** `cv2.Canny(enhanced, 40, 120, apertureSize=3, L2gradient=True)`
- **Rationale:** Utilizing the Euclidean $L_2$ gradient norm $\sqrt{G_x^2 + G_y^2}$ rather than the $L_1$ approximation $|G_x| + |G_y|$ provides accurate circle edge localization.

---

## 5. Circular Hough Transform (CHT)
### Mathematical Formulation
A circle in Cartesian coordinates is defined by:
$$(x - x_c)^2 + (y - y_c)^2 = r^2$$
This can be expressed parametrically as:
$$x = x_c + r \cos(\theta), \quad y = y_c + r \sin(\theta)$$
For every edge point $(x, y)$, the set of possible center coordinates $(x_c, y_c)$ forms a cone in 3D parameter space $(x_c, y_c, r)$. When edge gradient directions $\theta$ are known from Canny, voting is restricted to lines along the gradient vector:
$$x_c = x - r \cos(\theta), \quad y_c = y - r \sin(\theta)$$
Accumulator cells receiving the highest votes represent true circle centers.
### Implementation Details
- **OpenCV Function:**
  ```python
  cv2.HoughCircles(
      denoised, cv2.HOUGH_GRADIENT,
      dp=1.2, minDist=40,
      param1=100, param2=38,
      minRadius=min_r, maxRadius=max_r
  )
  ```
- **Parameter Significance:**
  - `dp=1.2`: Inverse accumulator resolution ratio.
  - `minDist=40`: Minimum physical distance separating detected circle centers.
  - `param1=100`: High threshold passed internally to Canny.
  - `param2=38`: Accumulator voting threshold at detection stage. Higher avoids noise; lower detects faint rims.

---

## 6. Marker-Controlled Watershed Segmentation
### Mathematical Formulation
The gradient image is viewed as a topographic relief map where gray level represents altitude. Catchment basins represent objects, and watershed divide lines represent boundaries.
To prevent over-segmentation from local texture minima, markers are extracted via Euclidean Distance Transform:
$$D(p) = \min_{q \in \text{Background}} \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}$$
Thresholding distance peaks:
$$\text{Sure Foreground} = \{p \mid D(p) \ge \tau \cdot \max(D)\}$$
Connected components on the sure foreground assign a unique label to each coin seed marker. The watershed algorithm floods the topological relief from these seed markers until water from different markers meets at contact ridges (labeled $-1$).
### Implementation Details
- **OpenCV Functions:**
  ```python
  dist = cv2.distanceTransform(morph_mask, cv2.DIST_L2, 5)
  _, sure_fg = cv2.threshold(dist, 0.42 * dist.max(), 255, 0)
  cv2.watershed(bgr_image, markers)
  ```

---

## 7. Dimensionless Shape Circularity
### Mathematical Formulation
$$C = \frac{4\pi \cdot A}{P^2}$$
where $A$ is the contour enclosed area and $P$ is the contour arc length perimeter.
### Derivation
For an ideal continuous circle of radius $r$:
$$A = \pi r^2, \quad P = 2\pi r$$
$$C = \frac{4\pi (\pi r^2)}{(2\pi r)^2} = \frac{4\pi^2 r^2}{4\pi^2 r^2} \equiv 1.0$$
For non-circular shapes:
- Equilateral Triangle: $C = \frac{\pi \sqrt{3}}{9} \approx 0.604$
- Square: $C = \frac{\pi}{4} \approx 0.785$
- Touching Coin Clump (Figure-8): $C \approx 0.45 - 0.58$
Filtering by $C \ge 0.65$ eliminates clumps, background shapes, and fragmented edge noise.

---

## 8. Physical Scale Calibration & Perspective
### Mathematical Formulation
$$S = \frac{\text{Known Reference Diameter (mm)}}{\text{Measured Reference Diameter (px)}}$$
$$\text{Physical Diameter (mm)} = \text{Measured Diameter (px)} \cdot S$$
### Optical Perspective Limitations
This linear relationship assumes an orthographic or fronto-parallel pinhole camera projection where the optical camera axis $\vec{n}_{\text{cam}}$ is parallel to the coin surface normal $\vec{n}_{\text{coin}}$:
$$\cos(\theta_{\text{tilt}}) = \vec{n}_{\text{cam}} \cdot \vec{n}_{\text{coin}} = 1.0$$
If tilted by angle $\theta_{\text{tilt}}$, circular coins project as ellipses with aspect ratio $\cos(\theta_{\text{tilt}})$, under-estimating the minor axis diameter. Homography rectification is required for non-perpendicular angles.
