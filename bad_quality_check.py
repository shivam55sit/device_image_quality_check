import os
import shutil
import json
import pandas as pd
from datetime import datetime

# IMPORT YOUR EXISTING FUNCTION (NO REFACTOR)
from MainQualitycheck import mainqualitycheck


# ===================== CONFIG =====================
INPUT_DIR = r"C:\Users\shivam.prajapati\Desktop\tele_crop_images"
GOOD_DIR = "./good_quality"
BAD_DIR = "./bad_quality"
REPORT_DIR = "./reports"

os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ===================== GOOD / BAD RULE =====================
def is_good_image(result: dict) -> bool:
    """
    STRICT CLINICAL RULE:
    Only 'Good Quality' images are accepted
    """
    return result.get("overall_quality") == "Good Quality"


# ===================== BATCH PROCESS =====================
def batch_quality_check():

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\nFound {len(image_files)} images for quality assessment\n")

    all_results = []

    for idx, image_name in enumerate(image_files, start=1):
        image_path = os.path.join(INPUT_DIR, image_name)
        print(f"[{idx}/{len(image_files)}] Processing: {image_name}")

        try:
            result = mainqualitycheck(image_path)
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            result = {
                "image_path": image_path,
                "overall_quality": "Error",
                "error": str(e)
            }

        # --------- CLASSIFICATION ---------
        if is_good_image(result):
            final_label = "GOOD"
            shutil.copy(image_path, os.path.join(GOOD_DIR, image_name))
        else:
            final_label = "BAD"
            shutil.copy(image_path, os.path.join(BAD_DIR, image_name))

        # --------- FLATTEN RESULT ---------
        record = {
            "image_name": image_name,
            "overall_quality": result.get("overall_quality"),
            "quality_pattern": result.get("quality_pattern"),
            "final_label": final_label,
            "processing_time_sec": round(result.get("processing_time", 0), 3),
            "recommendations": "; ".join(result.get("recommendations", []))
        }

        all_results.append(record)

    # ===================== SAVE REPORTS =====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(REPORT_DIR, f"batch_quality_report_{timestamp}.csv")
    json_path = os.path.join(REPORT_DIR, f"batch_quality_report_{timestamp}.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n==================== SUMMARY ====================")
    print(df["final_label"].value_counts())
    print(f"\nCSV report saved to: {csv_path}")
    print(f"JSON report saved to: {json_path}")
    print("Batch quality assessment completed.")
    print("================================================")


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    batch_quality_check()
