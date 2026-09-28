#!/usr/bin/env python3
"""
prepare_kaggle_training_upload.py

Run this script from your wind_paper project folder.

Example:
    cd ~/Desktop/wind_paper
    python prepare_kaggle_training_upload.py

It will:
1. Collect only the files needed by Notebook 06 on Kaggle.
2. Preserve the expected folder structure.
3. Verify that every required file exists.
4. Create:
       kaggle_training_upload/
       kaggle_training_upload.zip

Upload kaggle_training_upload.zip to Kaggle.
"""

from pathlib import Path
import shutil
import sys


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def find_project_root() -> Path:
    """
    Prefer the current directory.
    If the script is run from elsewhere, also try the script's directory.
    """
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path.home() / "Desktop" / "wind_paper",
    ]

    for candidate in candidates:
        if (
            (candidate / "data").exists()
            and (candidate / "outputs").exists()
            and (candidate / "notebooks").exists()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the wind_paper project root.\n"
        "Run this script from ~/Desktop/wind_paper"
    )


def main():
    project_root = find_project_root()

    output_dir = project_root / "kaggle_training_upload"
    zip_path = project_root / "kaggle_training_upload.zip"

    print("=" * 72)
    print("KAGGLE TRAINING UPLOAD PREPARATION")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print()

    # ---------------------------------------------------------
    # Required files
    # ---------------------------------------------------------

    required_files = []

    # Notebook 03 outputs
    required_files.append(
        (
            project_root / "data" / "processed" / "study_grid.csv",
            Path("data/processed/study_grid.csv"),
        )
    )

    for year in range(2005, 2021):
        required_files.append(
            (
                project_root
                / "data"
                / "processed"
                / "vector_features"
                / f"{year}.nc",
                Path("data/processed/vector_features") / f"{year}.nc",
            )
        )

    required_files.append(
        (
            project_root
            / "outputs"
            / "tables"
            / "temporal_scale_schema.csv",
            Path("outputs/tables/temporal_scale_schema.csv"),
        )
    )

    required_files.append(
        (
            project_root
            / "outputs"
            / "tables"
            / "forecast_horizon_schema.csv",
            Path("outputs/tables/forecast_horizon_schema.csv"),
        )
    )

    # Notebook 04 outputs
    required_files.append(
        (
            project_root
            / "data"
            / "splits"
            / "vector"
            / "split_manifest.csv",
            Path("data/splits/vector/split_manifest.csv"),
        )
    )

    required_files.append(
        (
            project_root
            / "outputs"
            / "models"
            / "vector_history_scaler.joblib",
            Path("outputs/models/vector_history_scaler.joblib"),
        )
    )

    required_files.append(
        (
            project_root
            / "outputs"
            / "models"
            / "vector_target_scaler.joblib",
            Path("outputs/models/vector_target_scaler.joblib"),
        )
    )

    required_files.append(
        (
            project_root
            / "outputs"
            / "models"
            / "vector_spatial_scaler.joblib",
            Path("outputs/models/vector_spatial_scaler.joblib"),
        )
    )

    # Notebook 05 outputs
    required_files.append(
        (
            project_root
            / "data"
            / "processed"
            / "shamal"
            / "shamal_hourly_labels.nc",
            Path("data/processed/shamal/shamal_hourly_labels.nc"),
        )
    )

    required_files.append(
        (
            project_root
            / "outputs"
            / "tables"
            / "shamal_split_regime_summary.csv",
            Path("outputs/tables/shamal_split_regime_summary.csv"),
        )
    )

    # ---------------------------------------------------------
    # Verify all files first
    # ---------------------------------------------------------

    missing = [
        src
        for src, _ in required_files
        if not src.exists()
    ]

    if missing:
        print("ERROR: These required files are missing:\n")

        for path in missing:
            print(f"  - {path}")

        print(
            "\nRun Notebooks 03, 04 and 05 successfully "
            "before creating the Kaggle upload."
        )

        sys.exit(1)

    print(f"All {len(required_files)} required files were found.")
    print()

    # ---------------------------------------------------------
    # Remove previous bundle
    # ---------------------------------------------------------

    if output_dir.exists():
        print(f"Removing old folder: {output_dir.name}")
        shutil.rmtree(output_dir)

    if zip_path.exists():
        print(f"Removing old ZIP: {zip_path.name}")
        zip_path.unlink()

    print()

    # ---------------------------------------------------------
    # Copy files
    # ---------------------------------------------------------

    copied_bytes = 0

    for src, relative_dest in required_files:
        dest = output_dir / relative_dest

        dest.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            src,
            dest,
        )

        file_size = src.stat().st_size
        copied_bytes += file_size

        print(
            f"COPIED  {relative_dest}  "
            f"({human_size(file_size)})"
        )

    print()
    print("-" * 72)
    print(
        f"Total copied size: {human_size(copied_bytes)}"
    )
    print("-" * 72)
    print()

    # ---------------------------------------------------------
    # Write a small manifest
    # ---------------------------------------------------------

    manifest_path = output_dir / "UPLOAD_CONTENTS.txt"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Kaggle training upload for wind_paper\n\n"
        )

        f.write(
            "Training: 2005-2017\n"
        )

        f.write(
            "Validation: 2018-2019\n"
        )

        f.write(
            "Reserved final test: 2020\n\n"
        )

        f.write(
            "Files included:\n"
        )

        for _, relative_dest in required_files:
            f.write(
                f"- {relative_dest.as_posix()}\n"
            )

    print(
        f"Created manifest: {manifest_path.relative_to(project_root)}"
    )

    # ---------------------------------------------------------
    # Create ZIP with Python standard library
    # ---------------------------------------------------------

    print()
    print("Creating ZIP file...")

    archive_base = project_root / "kaggle_training_upload"

    created_zip = Path(
        shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=str(project_root),
            base_dir="kaggle_training_upload",
        )
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Folder:\n  {output_dir}"
    )

    print()

    print(
        f"ZIP to upload to Kaggle:\n  {created_zip}"
    )

    print()

    print(
        f"ZIP size:\n  {human_size(created_zip.stat().st_size)}"
    )

    print()
    print(
        "Upload ONLY this file to Kaggle:"
    )

    print(
        f"  {created_zip.name}"
    )

    print()
    print(
        "Do NOT upload the old wind_kaggle_bundle.zip."
    )


if __name__ == "__main__":
    main()
