import sys
sys.path.append('geometry')

import numpy as np
import json
from loader import load_aneurisk_as_vessel_geometry
from morphology import MorphologyExtractor
from labels import load_aneurisk_location_risk


def build_aneurisk_labelled_dataset(aneurisk_dir):
    """
    Builds the labelled feature dataset for AneuRisk65: morphology
    features (from the PCA-slicing extractor - the only option here,
    since AneuRisk65 has no true surface mesh for VMTK) paired with
    LOCATION-RISK labels (not rupture - see labels.py caveat).
    """
    patients = load_aneurisk_location_risk(f'{aneurisk_dir}/Patients.txt')

    dataset = []
    skipped = []

    for p in patients:
        if not p["has_aneurysm"]:
            continue  # N-group has no aneurysm to extract morphology from

        case_file = f'{aneurisk_dir}/Rawdata_FKS_{p["patient_index"]}.txt'
        try:
            vg = load_aneurisk_as_vessel_geometry(case_file, case_id=p["code"])
            extractor = MorphologyExtractor(vg, n_slices=150)
            features = extractor.extract_all()
        except Exception as e:
            skipped.append({"code": p["code"], "reason": str(e)})
            continue

        # Plausibility check: real intracranial aneurysm aspect ratios
        # essentially never exceed ~5 in the clinical literature. The
        # PCA-slicing neck-detection can fail on individual cases
        # (especially reconstructed, non-scanned AneuRisk65 geometry),
        # producing physically impossible outliers. Flag rather than
        # silently pass through - the record's OTHER features (parent
        # diameter, curvature, undulation, ellipticity) remain valid
        # regardless, since they do not depend on neck detection.
        ar = features.get("aspect_ratio")
        is_plausible = (ar is None) or (0 < ar <= 5)

        record = {
            "code": p["code"],
            "location_group": p["location_group"],
            "is_upper_location_risk": p["is_upper_location_risk"],
            "side": p["side"],
            "neck_dependent_features_reliable": is_plausible,
            **{k: v for k, v in features.items() if k not in ("case_id", "source")}
        }
        if not is_plausible:
            record["reliability_note"] = (
                f"aspect_ratio={ar:.2f} exceeds clinically plausible range (>5) - "
                f"neck/height/AR/SR/bottleneck values for this case are UNRELIABLE "
                f"due to PCA-slicing neck-detection failure. Other features unaffected."
            )
        dataset.append(record)

    return dataset, skipped


if __name__ == "__main__":
    dataset, skipped = build_aneurisk_labelled_dataset(
        'data/raw/aneurisk_download/AneuRisk65'
    )
    n_unreliable = sum(1 for d in dataset if not d["neck_dependent_features_reliable"])
    print(f"Built {len(dataset)} labelled records, skipped {len(skipped)}")
    print(f"Neck-dependent features UNRELIABLE (flagged, not removed) in {n_unreliable}/{len(dataset)} cases")
    if skipped:
        print("Skipped cases:", skipped[:5])
    print()
    print("Sample record:")
    print(json.dumps(dataset[0], indent=2, default=str))

    with open('aneurisk65_labelled_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    print("\nSaved to aneurisk65_labelled_dataset.json")
