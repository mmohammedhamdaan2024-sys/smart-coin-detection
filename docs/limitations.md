# Engineering Limitations and Numismatic Disclosures

This document details the scientific, physical, and practical limitations of the **Smart Coin Detection System**. A rigorous computer vision system must acknowledge the physical boundaries of optical digital image processing.

---

## 1. Planar Perspective Distortion
### The Physical Problem
Geometric measurements ($D_{\text{px}} = 2r$) and circularity calculations ($C = \frac{4\pi A}{P^2}$) assume an orthogonal or fronto-parallel viewing angle, where the camera sensor plane is parallel to the coin surface.
### Practical Impact
If a handheld camera is held at an oblique perspective angle ($\theta_{\text{tilt}} > 10^\circ$):
- Circular coins optically deform into ellipses.
- The minor axis contracts by $\cos(\theta_{\text{tilt}})$.
- Measured circularity drops below $0.85$, potentially causing false rejection.
- Diameter is under-estimated along the tilt direction.
### Mitigation & Recommendation
Users must position the camera lens directly perpendicular ($90^\circ$) to the flat surface. Non-perpendicular acquisition requires 4-point perspective planar homography calibration using a known rectangular target.

---

## 2. Weight and Mass Determination
### The Physical Problem
A 2D digital photograph captures optical reflectance (photons per pixel sensor), not gravitational mass.
### Why Area Cannot Measure Weight
1. **Thickness Differences:** Two coins with the exact same $24$ mm diameter can have different thicknesses ($1.5$ mm vs $2.5$ mm), resulting in completely different volumes.
2. **Density & Alloy Variation:** A pure copper coin has density $\rho \approx 8.96\text{ g/cm}^3$, whereas an aluminum-bronze coin has density $\rho \approx 7.6\text{ g/cm}^3$, and a zinc-core coin has $\rho \approx 7.14\text{ g/cm}^3$.
3. **Planchet Cavities:** Photographs cannot detect internal cavities or cladded layers.
### System Enforcement
The system **never** fabricates a weight measurement from pixel area. It reports either:
- The official published reference mass for a confidently identified coin type.
- A physical scale measurement entered by the user.
- Explicit status: *"Weight unavailable — identify coin or provide physical measurement"*.

---

## 3. Automated Condition Screening vs Professional Numismatic Grading
### The Numismatic Boundary
Professional coin grading (e.g. PCGS or NGC using the 70-point Sheldon Scale from Poor-1 to Mint State-70) evaluates criteria that are physically invisible to a standard digital camera:
1. **Cartwheel Luster:** The dynamic rotational reflection of microscopic die-flow lines under tilting light.
2. **High-Point Friction:** Micro-friction on the highest anatomical relief point of a portrait (e.g., hair curls, eagle breast feathers).
3. **Bag Marks & Strike Weakness:** Distinguishing between planchet strike weakness and circulation handling abrasions requires stereoscopic microscope inspection.
### System Screening Scope
The system implements **surface anomaly screening**:
- Detects visible high-contrast scratches, tarnish patches, and rim dents.
- Reports categories: *No obvious anomaly*, *Minor irregularities*, *Moderate irregularities*, *Significant irregularities*.
- Accompanied by mandatory disclaimer: *"Image-based condition analysis is an automated visual screening tool, not a professional coin-grading or authentication service."*

---

## 4. Heavy Stacking and Occlusion
### The Topological Boundary
When coins are stacked vertically on top of one another such that more than $50\%$ of the lower coin's boundary is hidden:
- The circular arc of the lower coin is broken.
- Distance transform peaks of the lower coin are obscured by the upper coin.
### System Behavior
The system detects the top unobstructed coin cleanly and flags the contact cluster as an overlapping region, alerting the user rather than outputting false phantom measurements.

---

## 5. Glare, Shadows, and Harsh Specular Highlights
### The Lighting Boundary
Metallic coins behave as specular reflectors (mirror-like surfaces). Direct un-diffused flash creates harsh overexposed white spots ($\text{Pixel Value} \ge 250$) and hard cast shadows.
### System Behavior
The quality assessment module detects saturated pixels ($> 8\%$) and alerts the user to high glare. Diffused ambient lighting is recommended for optimal inspection.
