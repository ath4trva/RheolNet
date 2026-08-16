import numpy as np


def load_aneurisk_location_risk(patients_txt_path):
    """
    Loads AneuRisk65's Patients.txt and returns LOCATION-based risk
    grouping, NOT rupture status - this dataset contains no recorded
    rupture outcome. Verified against Sangalli et al. AneuRisk65 papers:

        type == 'U' : aneurysm at/after ICA terminal bifurcation
                       (Upper group - anatomically higher-risk location)
        type == 'L' : aneurysm before ICA terminal bifurcation
                       (Lower group - anatomically lower-risk location)
        type == 'N' : no cerebral aneurysm (control group)

    Returns a dict keyed by patient 'code' (matches Rawdata_FKS_<N>.txt
    ordering via the 'Patient' index column, NOT the code itself).
    """
    records = []
    with open(patients_txt_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            patient_idx, code, ptype, an_abscissa, side = parts[0], parts[1], parts[2], parts[3], parts[4]
            records.append({
                "patient_index": int(patient_idx),
                "code": code,
                "location_group": ptype,
                "is_upper_location_risk": ptype == "U",
                "has_aneurysm": ptype in ("U", "L"),
                "aneurysm_abscissa": None if an_abscissa == "NA" else float(an_abscissa),
                "side": side,
                "label_type": "location_risk_proxy",
                "label_source": "AneuRisk65_Patients.txt",
                "label_caveat": "NOT a rupture outcome - anatomical location grouping only"
            })
    return records
