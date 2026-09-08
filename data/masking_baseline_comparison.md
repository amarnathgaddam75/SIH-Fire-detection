# Masking Baseline vs. GeoFlare: Surveillance Blind Spot Analysis

> **Presentation Key Takeaway / Executive Quote:**  
> *"A masking-only approach would have discarded 1,114 thermal sources entirely, including 617 flagged Critical or High risk by our system -- none of which would be visible under a mask-and-discard approach."*

---

## 1. Executive Summary

Existing operational fire monitoring systems (such as the Forest Survey of India's Forest Fire Alert System) typically employ a **static spatial mask** to filter out known industrial zones, power plants, and mining leases. While this prevents false alarms in forest fire statistics, **it treats all industrial thermal activity as benign background noise**.

In mixed industrial-wildland belts like Korba-Raigarh (Chhattisgarh), industrial flares, coal depot spontaneous combustion, and refinery blowouts frequently border human settlements and protected sal forests. Discarding these hotspots creates an unmonitored surveillance blind spot.

---

## 2. Quantitative Comparison Table

| Metric | Traditional Masking-Only Baseline | GeoFlare Multi-Factor Intelligence | Impact of GeoFlare |
| :--- | :---: | :---: | :--- |
| **Total Hotspots Evaluated** | 2,008 | 2,008 | 100% full-spectrum coverage |
| **Hotspots Retained & Monitored** | 894 (44.5%) | **2,008 (100.0%)** | **+1,114 active hotspots monitored** |
| **Hotspots Silently Discarded** | **1,114 (55.5%)** | **0 (0.0%)** | Complete elimination of spatial blind spots |
| **Critical Risk Sources Monitored** | **0** *(all masked out)* | **31 (100.0%)** | **+31 life-safety crisis events detected** |
| **High Risk Sources Monitored** | **0** *(all masked out)* | **586 (100.0%)** | **+586 persistent industrial hazards tracked** |
| **False Alarm Reduction Mechanism** | Blind deletion of proximity data | AI Typology (Industrial vs. Wildfire) | Eliminates false alarms without losing visibility |

---

## 3. Breakdown of the 1,114 Masked-Out Thermal Sources

Under GeoFlare's multi-factor risk assessment model, the 1,114 thermal sources discarded by a static 1 km industrial buffer break down into:

- **Critical Risk**: **31 hotspots (2.8%)**  
  Severe thermal surges (2.5x to 6.2x above normal cluster baseline) occurring within 2.5 km of residential neighborhoods (16 occurring directly within 500 m of homes).
- **High Risk**: **586 hotspots (52.6%)**  
  Active, high-heat industrial flare stacks, smelters, and kiln operations located in close proximity (<= 1.5 km) to population settlements.
- **Medium Risk**: **497 hotspots (44.6%)**  
  Moderate thermal sources and low-persistence edge fires situated at intermediate distances (1.5 km to 3 km) from communities.
- **Low Risk**: **0 hotspots (0.0%)**  
  Remote or isolated low-intensity thermal readings.

---

## 4. Key Case Studies Discarded by Simple Masking

Every one of the **31 Critical Risk** events in the entire Korba basin falls within the 1 km industrial perimeter. A masking-only product would have provided **zero alerts** for:

1. **Row 113 & 140 (March 4–5, 2026)**:
   - FRP: **15.28 MW** and **15.02 MW** (5.5x cluster baseline surge).
   - Proximity: **293 m to 365 m** from residential township colonies.
   - *Masking Result*: **Discarded**.
   - *GeoFlare Action*: Flagged **Critical Risk** with abnormal FRP spike notification.

2. **Row 56 & 1924 (March 2 & June 24, 2026)**:
   - Proximity: **0.0 m to 67 m** from mapped residential polygons.
   - *Masking Result*: **Discarded**.
   - *GeoFlare Action*: Flagged **Critical Risk** due to direct residential boundary contact.

---

## 5. Architectural Conclusion for Presentation

> *"Masking prevents false alarms by destroying data. GeoFlare prevents false alarms by understanding data."*

By replacing static spatial subtraction with **AI classification (XGBoost)**, **spatio-temporal clustering (DBSCAN)**, and **residential proximity risk scoring**, GeoFlare preserves complete surveillance over industrial disasters while maintaining zero false positive contamination in forest conservation reporting.
