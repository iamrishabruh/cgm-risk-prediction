# Clinical Limitations and Safety Posture

This software is a **machine learning engineering and research prototype**. It is **not** a medical device, not FDA-cleared or CE-marked, and **not** validated for clinical use.

## What this system does not do

- It does **not** diagnose diabetes or its complications.  
- It does **not** predict imminent hypoglycemia or hyperglycemia events.  
- It does **not** replace self-monitoring, laboratory HbA1c, or clinician judgment.  
- It does **not** account for medications, meals, illness, or insulin dosing.

## Model outputs are not deployment-grade

**Do not** describe checkpoints, dashboards, or APIs as “clinical deployment-ready.” The repository demonstrates **ML workflow patterns** (data prep, baselines, CV, service boundaries), not regulatory evidence.

## Data and fairness

- Performance may differ across age groups, sex/gender, ethnicity, insulin modality, CGM vendor, and socioeconomic contexts. **Subgroup analyses are not implemented** in the default scripts.  
- Missing data handling uses **simple heuristics** (e.g., column means). This can encode biases present in the training table.

## Human oversight

Any future use in patient-facing settings would require:

- Prospective protocol design with a qualified clinical team.  
- Independent validation on held-out sites and time periods.  
- Risk management for incorrect predictions (false reassurance or unnecessary alarm).  
- Compliance with local regulations (e.g., medical device rules, privacy frameworks).

## Incident reporting

If you adapt this code in a regulated environment, maintain your own adverse-event and model monitoring processes. This open-source repository provides **no operational guarantees**.
