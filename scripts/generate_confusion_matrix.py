"""Generate confusion matrices for a given YOLO model and target zone."""
import argparse
import tempfile
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt weights")
    parser.add_argument("--zone", required=True, choices=["CZ_A", "CZ_B", "CZ_C"], help="Target climate zone")
    parser.add_argument("--data_dir", default="data", help="Directory of dataset")
    parser.add_argument("--project", default="results/confusion_matrices", help="Output project directory")
    parser.add_argument("--name", required=True, help="Name of directory for this run")
    parser.add_argument("--device", default="", help="CPU or GPU device")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    zone = args.zone
    
    # Create temp YAML config
    tmp_yaml = Path(tempfile.gettempdir()) / f"eval_cm_{zone}.yaml"
    zone_data = {
        "path": str(data_dir.parent),  # project root
        "train": f"data/{zone}/images/test",  # placeholder to satisfy validation check
        "val":   f"data/{zone}/images/test",
        "nc":    8,
        "names": [
            "Building", "Small Car", "Truck", "Bus",
            "Cargo Truck", "Shipping Container", "Vehicle Lot", "Shed"
        ]
    }
    
    with open(tmp_yaml, "w") as f:
        yaml.dump(zone_data, f)
        
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    print(f"Running validation on {zone} to compute confusion matrix...")
    model.val(
        data=str(tmp_yaml),
        imgsz=640,
        batch=16,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name,
        plots=True,      # This ensures confusion_matrix.png is generated
        save_json=False,
        verbose=True
    )
    print(f"✓ Confusion matrix saved to: {Path(args.project) / args.name}")

if __name__ == "__main__":
    main()
