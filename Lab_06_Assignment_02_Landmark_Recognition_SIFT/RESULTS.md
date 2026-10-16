# Landmark Recognition Using SIFT – Results

*Generated automatically by `Landmark_Recognition_SIFT.ipynb` (OpenCV 4.13.0).*

> **Please read:** the sandbox used to build this submission had no internet access, so the landmark images in `dataset/` and `queries/` are **procedurally rendered illustrations** (see `build_database.py`), not real photographs. The pipeline is identical for real photos – replace the images and re-run the notebook; every number in this file will be regenerated.

## 1. Landmark image database (Task 1)

| Landmark | Folder | Images | Views |
|---|---|---|---|
| Colosseum | dataset/Colosseum/ | 4 | far_away, front_close, left_angle, right_angle |
| Eiffel Tower | dataset/Eiffel_Tower/ | 4 | far_away, front_close, left_angle, right_angle |
| Great Wall of China | dataset/Great_Wall_of_China/ | 4 | far_away, front_close, left_angle, right_angle |
| Statue of Liberty | dataset/Statue_of_Liberty/ | 4 | far_away, front_close, left_angle, right_angle |
| Taj Mahal | dataset/Taj_Mahal/ | 4 | far_away, front_close, left_angle, right_angle |

![database](screenshots/01_database.png)

## 2. SIFT feature extraction (Task 2)

Grayscale → `cv2.SIFT_create()` → keypoints + 128-D descriptors, stored with their landmark label in `outputs/sift_database.pkl`. Database feature extraction took **2.35 s**.

| Landmark | Images | Total keypoints | Keypoints per image |
|---|---|---|---|
| Colosseum | 4 | 6451 | 932, 2242, 1858, 1419 |
| Eiffel Tower | 4 | 3469 | 467, 1071, 925, 1006 |
| Great Wall of China | 4 | 5097 | 594, 2002, 1463, 1038 |
| Statue of Liberty | 4 | 4009 | 575, 1315, 1013, 1106 |
| Taj Mahal | 4 | 7092 | 1203, 2043, 2000, 1846 |

![reference keypoints](screenshots/02_reference_keypoints.png)

## 3. Query image processing (Task 3)

![query keypoints](screenshots/03_query_keypoints.png)

## 4. Matching and recognition (Task 4)

* Brute-Force matcher, Euclidean (L2) distance, kNN with k = 2
* Lowe's ratio test: keep a match if `d1 < 0.75 × d2`
* Landmarks ranked by reliable matches (best reference view)
* Rejection rule – a landmark is accepted only if **all** hold:

| Parameter | Value | Meaning |
|---|---|---|
| `RATIO` | 0.75 | Lowe's ratio threshold |
| `MIN_GOOD` | 25 | minimum reliable matches |
| `MARGIN` | 1.5 | winner must have ≥ this × the runner-up's matches |
| `MIN_INLIERS` | 15 | minimum RANSAC inliers |
| `RANSAC_THR` | 5.0 px | RANSAC reprojection threshold |

Example ranking (Eiffel Tower query, held-out photo):

| Rank | Landmark | Reliable matches (best view) | Total over all views | Best reference view |
|---|---|---|---|---|
| 1 | Eiffel Tower | 160 | 627 | front_close |
| 2 | Statue of Liberty | 14 | 34 | right_angle |
| 3 | Great Wall of China | 13 | 45 | front_close |
| 4 | Colosseum | 10 | 24 | front_close |
| 5 | Taj Mahal | 7 | 24 | front_close |

![ranking](screenshots/04_ranking.png)

## 5. Landmark localization (Task 5)

RANSAC homography (reference → query), the four reference corners are projected and a polygon is drawn only when geometric verification passes (≥ 15 inliers, convex polygon, plausible size/scale). Otherwise a message is shown and no polygon is drawn.

![full result](screenshots/05_full_result_demo.png)

## 6. Performance evaluation (Task 6)

### 6.1 Required test conditions

| Ground truth | Condition | Best candidate | Predicted landmark | Keypoints | Reliable matches | RANSAC inliers | Decision | Correct | Time (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Eiffel Tower | Original | Eiffel Tower | Eiffel Tower | 1812 | 158 | 112 | Recognized | ✔ | 1257 |
| Eiffel Tower | Rotated (45°) | Eiffel Tower | Eiffel Tower | 3453 | 396 | 106 | Recognized | ✔ | 2534 |
| Eiffel Tower | Resized (50%) | Eiffel Tower | Eiffel Tower | 349 | 54 | 45 | Recognized | ✔ | 226 |
| Eiffel Tower | Reduced brightness (x0.4) | Eiffel Tower | Eiffel Tower | 261 | 97 | 68 | Recognized | ✔ | 253 |
| Eiffel Tower | Different viewpoint | Eiffel Tower | Eiffel Tower | 1456 | 126 | 93 | Recognized | ✔ | 996 |
| Not in database | Unknown landmark – original | Eiffel Tower | — (rejected) | 3608 | 23 | 6 | Rejected | ✔ | 2128 |

Screenshots of these six tests:

**Eiffel Tower – Original**

![test 7](screenshots/06_required_test_07.png)

**Eiffel Tower – Rotated (45°)**

![test 8](screenshots/06_required_test_08.png)

**Eiffel Tower – Resized (50%)**

![test 9](screenshots/06_required_test_09.png)

**Eiffel Tower – Reduced brightness (x0.4)**

![test 10](screenshots/06_required_test_10.png)

**Eiffel Tower – Different viewpoint**

![test 11](screenshots/06_required_test_11.png)

**Pyramids of Giza – Unknown landmark – original**

![test 31](screenshots/06_required_test_31.png)


### 6.2 Extended evaluation – all landmarks, all conditions (34 images)

| Ground truth | Condition | Best candidate | Predicted landmark | Keypoints | Reliable matches | RANSAC inliers | Decision | Correct | Time (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Colosseum | Original | Colosseum | Colosseum | 3018 | 634 | 572 | Recognized | ✔ | 1931 |
| Colosseum | Rotated (45°) | Colosseum | Colosseum | 5511 | 781 | 559 | Recognized | ✔ | 3840 |
| Colosseum | Resized (50%) | Colosseum | Colosseum | 737 | 217 | 180 | Recognized | ✔ | 584 |
| Colosseum | Reduced brightness (x0.4) | Colosseum | Colosseum | 962 | 428 | 377 | Recognized | ✔ | 787 |
| Colosseum | Different viewpoint | Colosseum | Colosseum | 2802 | 597 | 532 | Recognized | ✔ | 1652 |
| Colosseum | Combined (rot 30° + 60% + dark) | Colosseum | Colosseum | 1177 | 275 | 192 | Recognized | ✔ | 777 |
| Eiffel Tower | Original | Eiffel Tower | Eiffel Tower | 1812 | 158 | 112 | Recognized | ✔ | 1257 |
| Eiffel Tower | Rotated (45°) | Eiffel Tower | Eiffel Tower | 3453 | 396 | 106 | Recognized | ✔ | 2534 |
| Eiffel Tower | Resized (50%) | Eiffel Tower | Eiffel Tower | 349 | 54 | 45 | Recognized | ✔ | 226 |
| Eiffel Tower | Reduced brightness (x0.4) | Eiffel Tower | Eiffel Tower | 261 | 97 | 68 | Recognized | ✔ | 253 |
| Eiffel Tower | Different viewpoint | Eiffel Tower | Eiffel Tower | 1456 | 126 | 93 | Recognized | ✔ | 996 |
| Eiffel Tower | Combined (rot 30° + 60% + dark) | Eiffel Tower | Eiffel Tower | 362 | 97 | 69 | Recognized | ✔ | 276 |
| Great Wall of China | Original | Great Wall of China | Great Wall of China | 2604 | 356 | 324 | Recognized | ✔ | 1627 |
| Great Wall of China | Rotated (45°) | Great Wall of China | Great Wall of China | 4923 | 444 | 266 | Recognized | ✔ | 3611 |
| Great Wall of China | Resized (50%) | Great Wall of China | Great Wall of China | 655 | 188 | 176 | Recognized | ✔ | 506 |
| Great Wall of China | Reduced brightness (x0.4) | Great Wall of China | Great Wall of China | 243 | 55 | 48 | Recognized | ✔ | 227 |
| Great Wall of China | Different viewpoint | Great Wall of China | Great Wall of China | 2082 | 277 | 246 | Recognized | ✔ | 1898 |
| Great Wall of China | Combined (rot 30° + 60% + dark) | Great Wall of China | Great Wall of China | 495 | 98 | 72 | Recognized | ✔ | 381 |
| Statue of Liberty | Original | Statue of Liberty | Statue of Liberty | 1898 | 391 | 369 | Recognized | ✔ | 1287 |
| Statue of Liberty | Rotated (45°) | Statue of Liberty | Statue of Liberty | 4069 | 492 | 390 | Recognized | ✔ | 2980 |
| Statue of Liberty | Resized (50%) | Statue of Liberty | Statue of Liberty | 426 | 91 | 85 | Recognized | ✔ | 300 |
| Statue of Liberty | Reduced brightness (x0.4) | Statue of Liberty | Statue of Liberty | 184 | 27 | 22 | Recognized | ✔ | 193 |
| Statue of Liberty | Different viewpoint | Statue of Liberty | Statue of Liberty | 1424 | 220 | 211 | Recognized | ✔ | 1157 |
| Statue of Liberty | Combined (rot 30° + 60% + dark) | Statue of Liberty | Statue of Liberty | 364 | 47 | 33 | Recognized | ✔ | 252 |
| Taj Mahal | Original | Taj Mahal | Taj Mahal | 2887 | 457 | 393 | Recognized | ✔ | 1856 |
| Taj Mahal | Rotated (45°) | Taj Mahal | Taj Mahal | 5534 | 583 | 381 | Recognized | ✔ | 3581 |
| Taj Mahal | Resized (50%) | Taj Mahal | Taj Mahal | 888 | 169 | 135 | Recognized | ✔ | 542 |
| Taj Mahal | Reduced brightness (x0.4) | Taj Mahal | Taj Mahal | 985 | 234 | 196 | Recognized | ✔ | 777 |
| Taj Mahal | Different viewpoint | Taj Mahal | Taj Mahal | 2166 | 373 | 310 | Recognized | ✔ | 1368 |
| Taj Mahal | Combined (rot 30° + 60% + dark) | Taj Mahal | Taj Mahal | 1316 | 228 | 142 | Recognized | ✔ | 761 |
| Not in database | Unknown landmark – original | Eiffel Tower | — (rejected) | 3608 | 23 | 6 | Rejected | ✔ | 2128 |
| Not in database | Unknown landmark – rotated (45°) | Eiffel Tower | — (rejected) | 6651 | 82 | 7 | Rejected | ✔ | 4245 |
| Not in database | Unknown landmark – different viewpoint | Colosseum | — (rejected) | 3079 | 14 | 5 | Rejected | ✔ | 2199 |
| Not in database | Unknown – scene without any landmark | Eiffel Tower | — (rejected) | 1263 | 15 | 5 | Rejected | ✔ | 888 |

**Overall accuracy: 100.0%** (34/34).  Screenshots of every test are in `screenshots/evaluation/`.

### 6.3 Summary per condition (known landmarks)

| Condition | Tests | Recognized | Mean keypoints | Mean reliable matches | Min inliers | Mean inliers | Mean time (ms) |
|---|---|---|---|---|---|---|---|
| Original | 5 | 5 | 2444 | 399 | 112 | 354 | 1592 |
| Rotated (45°) | 5 | 5 | 4698 | 539 | 106 | 340 | 3309 |
| Resized (50%) | 5 | 5 | 611 | 144 | 45 | 124 | 432 |
| Reduced brightness (x0.4) | 5 | 5 | 527 | 168 | 22 | 142 | 447 |
| Different viewpoint | 5 | 5 | 1986 | 319 | 93 | 278 | 1414 |
| Combined (rot 30° + 60% + dark) | 5 | 5 | 743 | 149 | 33 | 102 | 489 |

### 6.4 Why unknown images were rejected

| Test image | Best (wrong) candidate | Reliable matches | RANSAC inliers | Rejected because |
|---|---|---|---|---|
| Unknown landmark – original | Eiffel Tower | 23 | 6 | too few reliable matches (23 < 25); ambiguous: best 23 vs runner-up 21 (< 1.5x); geometric verification failed: only 6 RANSAC inliers (need >= 15) |
| Unknown landmark – rotated (45°) | Eiffel Tower | 82 | 7 | ambiguous: best 82 vs runner-up 72 (< 1.5x); geometric verification failed: only 7 RANSAC inliers (need >= 15) |
| Unknown landmark – different viewpoint | Colosseum | 14 | 5 | too few reliable matches (14 < 25); ambiguous: best 14 vs runner-up 14 (< 1.5x); geometric verification failed: only 5 RANSAC inliers (need >= 15) |
| Unknown – scene without any landmark | Eiffel Tower | 15 | 5 | too few reliable matches (15 < 25); ambiguous: best 15 vs runner-up 12 (< 1.5x); geometric verification failed: only 5 RANSAC inliers (need >= 15) |

## 7. Discussion


### Overall performance
* **30/30** landmark queries (5 landmarks × 6 conditions) were recognized with the correct name and **4/4** images that are *not* in the database were rejected → **100.0% overall accuracy** on 34 test images.
* Every correct detection had at least **27** reliable matches and **22** RANSAC inliers, while the unknown images never got more than **7** inliers. That clear gap is what makes a single fixed threshold (`MIN_INLIERS = 15`) work.
* Average processing time was **1408 ms** per query on CPU (feature extraction + brute-force matching against 20 reference images + RANSAC).

### Effect of each condition (Eiffel Tower example)
* **Rotation (45°)** – SIFT assigns every keypoint a dominant orientation, so it is rotation-invariant: 396 reliable matches / 106 inliers vs 158 / 112 for the original. The larger canvas simply produced more keypoints, so processing time grew (2534 ms vs 1257 ms).
* **Resizing (50%)** – matches dropped from 158 to 54: scale-space detection copes with the size change, but half the pixels means the finest-scale details disappear. Still 45 inliers, so recognition and localization succeed.
* **Reduced brightness (×0.4)** – lowering contrast pushes many keypoints below SIFT's contrast threshold (Eiffel: 1812 → 261 keypoints) yet 68 inliers remained. SIFT descriptors are normalised, hence illumination-robust, but the loss of keypoints is the real cost.
* **Different viewpoint** – 126 reliable matches / 93 inliers. The homography absorbs the perspective change because the landmark is (nearly) planar in these images.
* **Weakest case overall:** Statue of Liberty – Reduced brightness (x0.4) with only 27 reliable matches / 22 inliers, i.e. close to the acceptance thresholds (25 / 15). Stacking several degradations (small + dark + rotated) is the most likely place for this system to start failing.

### Why the match count alone is not enough
The unknown image "*Unknown landmark – rotated (45°)*" obtained **82 reliable matches** with "Eiffel Tower" – more than `MIN_GOOD` – but only **7 RANSAC inliers**. Those matches are scattered randomly (texture look-alikes), so no consistent homography exists and geometric verification rejects it. Without the RANSAC check this image would have been wrongly "recognized". The runner-up margin test (`MARGIN = 1.5`) adds protection against ambiguous cases.

### Limitations
1. **Synthetic database.** The images in this submission were rendered procedurally because the build machine had no internet. They contain rich texture and identical surface patterns across views, which is *easier* than real tourist photos. Real photos (crowds, weather, trees, partial occlusion, different seasons, day vs. night) will give fewer matches, and the clean gap between correct and false inliers will shrink – thresholds must be re-tuned on real data.
2. **Planar assumption.** A homography only models a planar object (or a distant one). A real 3-D landmark seen from a very different side (e.g. front vs. back of the Taj Mahal) is not described by one homography, so inliers drop even when it is the right landmark.
3. **Speed / scalability.** Brute-force matching is O(N·M) per image pair, so it will become slow with thousands of reference images; the approximate FLANN k-d tree matcher or a bag-of-visual-words index is needed for a real tourism system. 1408 ms/query is fine for 20 images only.
4. **Parameters are hand-tuned** (`RATIO`, `MIN_GOOD`, `MARGIN`, `MIN_INLIERS`, `RANSAC_THR`) on a small test set; they are not guaranteed to generalise.
5. **Low contrast / very small objects** reduce the number of keypoints sharply (see brightness test). Pre-processing such as CLAHE, or lowering SIFT's `contrastThreshold`, could help.
6. **One landmark per query** – the pipeline reports only the single best landmark and one polygon; several landmarks in one photo are not handled.
7. **SIFT ignores colour**, and the reference crop used for the polygon includes a small margin of background, so the polygon is slightly larger than the landmark.

### Possible improvements
More reference views per landmark, FLANN matching, RootSIFT descriptors, CLAHE for dark images, multi-instance RANSAC for several landmarks, and learned descriptors (e.g. SuperPoint/LightGlue) for harder, real-world viewpoint changes.


## 8. Files

| Path | Content |
|---|---|
| `Landmark_Recognition_SIFT.ipynb` | main notebook (already executed, outputs included) |
| `landmark_recognition.py` | the same code exported as a plain Python script |
| `build_database.py` | generator of the (synthetic) image database and query images |
| `dataset/<Landmark>/` | reference images (5 landmarks × 4 views) |
| `queries/` | held-out and transformed test images |
| `screenshots/` | all figures / application output screenshots |
| `outputs/evaluation_results.csv` | evaluation table as CSV |
| `outputs/sift_database.pkl` | stored descriptors + labels |
