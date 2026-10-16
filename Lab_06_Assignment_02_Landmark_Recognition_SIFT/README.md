# SIFT Landmark Recognition – Lab Assignment

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Landmark_Recognition_SIFT.ipynb      # Kernel -> Restart & Run All
```
Run it from this folder (all paths are relative). Running the notebook regenerates `RESULTS.md`, `outputs/` and `screenshots/`.
Plain-script alternative: `python landmark_recognition.py`.

## Folder layout
| Path | What it is |
|---|---|
| `Landmark_Recognition_SIFT.ipynb` | Main deliverable: Tasks 1-6, already executed (outputs visible) |
| `RESULTS.md` | All results: tables, screenshots, discussion, limitations |
| `landmark_recognition.py` | Same code as the notebook, as a script |
| `build_database.py` | Creates the image database + query images |
| `dataset/<Landmark>/` | Reference images – 5 landmarks x 4 views |
| `queries/` | Held-out query photos and the transformed test images |
| `screenshots/` | Application output screenshots (`evaluation/` = every test) |
| `outputs/` | `evaluation_results.csv`, `sift_database.pkl` (descriptors + labels) |

## IMPORTANT: images are synthetic
The environment this was built in had no internet access, so the landmark images are procedurally rendered illustrations (not real photos). To use real photos:
1. Delete the contents of `dataset/` and `queries/`.
2. Put real reference photos in `dataset/Eiffel_Tower/`, `dataset/Taj_Mahal/`, ... (any number of images per folder, at least 5 landmarks).
3. For each landmark put a **held-out** photo at `queries/<Landmark>/original.jpg` and a different-angle photo at `queries/<Landmark>/viewpoint.jpg`; the notebook creates the rotated / resized / dark versions itself. Put any non-database landmark in `queries/Pyramids_of_Giza/` (original.jpg + viewpoint.jpg) for the "unknown" test.
4. Re-run the notebook. Thresholds (`MIN_GOOD`, `MARGIN`, `MIN_INLIERS`) are in the config cell – re-tune them on real photos.
