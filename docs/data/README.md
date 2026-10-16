# Reference Data Directory

This directory stores coin reference information and sample images for calibration, classification, and testing.

## Files
- `coin_reference.json`: The core local reference database containing physical dimensions (diameter in mm, thickness in mm, mass in grams), materials, issue periods, and mint information for standard demonstration currencies (USD, EUR, GBP, INR).
- `sample_images/`: Test images with known coin arrangements (single coin, multiple separated, touching, overlapping, uncalibrated).

## Schema for Coin Reference Records
```json
{
  "reference_id": "STRING (Unique identifier, e.g., USD_QUARTER)",
  "currency": "STRING (ISO currency code, e.g., USD)",
  "currency_symbol": "STRING (e.g., $)",
  "country": "STRING (Issuing authority)",
  "denomination": "STRING (Human readable denomination)",
  "face_value": "FLOAT (Numerical face value)",
  "reference_diameter_mm": "FLOAT (Official nominal diameter in millimeters)",
  "reference_thickness_mm": "FLOAT (Official nominal thickness in millimeters)",
  "reference_mass_g": "FLOAT (Standard mass in grams)",
  "material": "STRING (Metallurgical composition)",
  "color_class": "STRING (copper_red | silver_nickel | brass_gold | bimetallic)",
  "edge_type": "STRING (Plain, Reeded, Scalloped, etc.)",
  "issue_period": "STRING (Historical or active minting period)",
  "obverse_desc": "STRING (Visual description)",
  "reverse_desc": "STRING (Visual description)",
  "source": "STRING (Documented authority or mint publication)"
}
```

## Adding New Reference Coins
Users can append additional custom reference coins directly through the Streamlit UI under **Settings / Reference Database** or by editing `coin_reference.json`.
