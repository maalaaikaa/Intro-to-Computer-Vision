# Skin Lesion Boundary Detection Using Canny Edge Detection

## Objective
Detect approximate skin lesion boundaries using filtering, Canny edge detection, morphology and contour extraction. Compare how effectively these methods separate the lesion from surrounding skin.

## Dataset
Dataset: [Skin Cancer ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).

Download identifier: `nodoubttome/skin-cancer9-classesisic`. Training classes: melanoma, pigmented benign keratosis, basal cell carcinoma and nevus. Five fixed images are used; no classifier is trained.

## Processing parameters
| Setting | Value |
| --- | --- |
| Resize | Longest side 512 pixels; original aspect ratio preserved |
| Main filter | Gaussian, 5 x 5 kernel |
| Canny comparisons | 50-100; 100-200; 150-250; additional 20-40 |
| Selected Canny pair | 20-40, selected from visual comparison |
| Morphological closing | 5 x 5 kernel, 2 iterations |
| Contour rule | Largest eligible external contour, away from the frame; area 1%-85% of image |
| Fallback | Inverse Otsu on Gaussian image, opening and closing; explicitly labelled |
| Area | Nonzero pixels in the filled contour mask, including its boundary |
| Perimeter | Closed contour arc length in pixels |
| Eight-method comparison | Original / Average / Gaussian / Median with Sobel / Canny |
| Sobel display | Normalized gradient magnitude with Otsu binarization |
| Comparison Canny | 20-40 for every filter |

## Tasks 1 to 6
1. Load each original image and resize it for processing.
2. Convert to grayscale and apply Gaussian smoothing.
3. Display the three example Canny settings and the additional lower setting.
4. Select 20-40 and explain the image-specific reasons below.
5. Close edge gaps and draw a candidate contour; label thresholding fallbacks.
6. Fill the contour, count its pixels and calculate perimeter.

## Required results table
| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| --- | --- | --- | --- | --- |
| Image 1 | Gaussian | Canny 20-40 | 44208 | 885.91 |
| Image 2 | Gaussian | Canny 20-40 | 15304 | 1760.15 |
| Image 3 | Gaussian | Canny 20-40 | 36459 | 1304.97 |
| Image 4 | Gaussian | Canny 20-40 | 48467 | 977.67 |
| Image 5 | Gaussian | Canny 20-40 | 29172 | 1127.85 |

## Image identities and boundary sources
| Image | Filename | Class | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) | Boundary source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Image 1 | ISIC_0000139.jpg | melanoma | Gaussian | Canny 20-40 | 44208 | 885.91 | Otsu fallback + contour |
| Image 2 | ISIC_0024435.jpg | pigmented benign keratosis | Gaussian | Canny 20-40 | 15304 | 1760.15 | Canny + closing + contour |
| Image 3 | ISIC_0024504.jpg | basal cell carcinoma | Gaussian | Canny 20-40 | 36459 | 1304.97 | Otsu fallback + contour |
| Image 4 | ISIC_0000019.jpg | nevus | Gaussian | Canny 20-40 | 48467 | 977.67 | Otsu fallback + contour |
| Image 5 | ISIC_0000141.jpg | melanoma | Gaussian | Canny 20-40 | 29172 | 1127.85 | Canny + closing + contour |

**Measurement note:** Values refer to resized candidate regions, not physical lesion dimensions. Images using Otsu are not Canny-only results. Image 3 is a poor segmentation; its measured region includes surrounding skin.

## Final comparison table
| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
| --- | --- | --- | --- | --- |
| Original + Sobel | Weak; texture or strong artifacts dominate | Variable: thick responses or very sparse edges | Misses weak outlines; almost blank for Image 4 | Poor overall on these images |
| Original + Canny | Weak; substantial background clutter | Thin edges but many unrelated responses | Lesion texture, hair and skin edges compete | Too cluttered for direct boundary selection |
| Average + Sobel | Reduces fine variation; strong hairs remain | Broad gradient bands | Shows lesion regions but thick boundaries | Moderate; limited precision |
| Average + Canny | Strong reduction of fine texture | Thin and sparse; faint details also lost | Cleaner maps but several incomplete outlines | Useful cleaner alternative |
| Gaussian + Sobel | Reduces texture; retains strong hair | Broad or broken responses | Some margins visible; hairs remain prominent | Moderate |
| Gaussian + Canny | Less clutter than unfiltered Canny | Thin; retains more weak detail than averaging | Edge contours usable for Images 2 and 5; others need fallback | Selected balance; not reliable on all images |
| Median + Sobel | Fine texture reduced; artifacts remain | Variable; sparse or almost blank in Image 4 | Weak margins may disappear after binarization | Inconsistent |
| Median + Canny | Less clutter than unfiltered Canny | Thin; internal details and hairs remain | Reasonable outlines in some images; gaps persist | Useful alternative; still needs region processing |

These qualitative observations describe the five displayed images. Noise handling means suppression of fine texture and edge clutter; no quantitative robustness or expert-mask accuracy test is claimed.


## Task 4 explanation and observations for each image

| Image | Why 20–40 was selected | Boundary result and limitation |
|---|---|---|
| 1 — melanoma | 50–100 retains mainly the prominent hair and a few lesion fragments; higher settings lose almost all lesion detail. 20–40 retains much more of the lesion margin. | Edge contours do not close adequately. Otsu provides a central region; some pale outer pigmentation is omitted. |
| 2 — pigmented benign keratosis | The lesion is faint. 50–100 gives very few fragments; the two higher settings are nearly empty. 20–40 retains its central texture and outline clues. | Closing yields an irregular edge-derived contour. It includes some nearby texture and misses parts of the faint margin; the area and especially the perimeter are rough estimates. |
| 3 — basal cell carcinoma | All settings struggle. 20–40 provides the most lesion-region information, but also many hair edges. 100–200 and 150–250 mainly retain the small dark spot and a few strong lines. | The Otsu fallback includes surrounding skin and misses portions of the visible lesion. This is a **poor boundary estimate**, not successful lesion separation. |
| 4 — nevus | 50–100 gives parts of the outline and strong hairs. The two higher settings lose nearly all edges. 20–40 retains a more complete but textured lesion region. | Otsu isolates the darker central region after edge contour extraction fails. The pale peripheral margin is partly excluded. |
| 5 — melanoma | All three example settings are almost empty. 20–40 reveals the main lesion structure with little distant background clutter. | Closing produces an edge-derived contour; it includes a small extension on the right and omits some faint pigmentation. |

**Best filter interpretation:** Gaussian is selected as the best practical compromise for this five-image lab because it reduces fine texture while preserving weak details. Average + Canny is cleaner in several images but also removes more faint information. This is a qualitative choice for these examples, not a universal or statistically proven ranking.

## Answers to the six questions

**1. Why is Gaussian filtering applied before Canny detection?**  
Gaussian filtering smooths small intensity fluctuations so they are less likely to produce false edges. In these results, filtered Canny maps have much less fine background clutter than unfiltered Canny. Strong hairs remain, and excessive smoothing could erase the faint lesion margin.

**2. How did the three Canny threshold settings affect the result?**  
50–100 retained more edge fragments than 100–200 and 150–250, but it still lost much of the lesion outline. In Images 2, 4 and 5 the higher settings were almost empty. Image 1 retained part of a strong hair at 100–200; Image 3 retained a small dark spot and a few strong lines at high thresholds. Thus fewer edges did not mean a better lesion boundary.

**3. Which threshold produced the best lesion boundary?**  
Among the three example pairs, 50–100 retained the most lesion information, although no pair produced a reliable complete outline for all five images. The additional 20–40 setting was selected because it preserved weak lesion edges that the higher pairs missed. It enabled usable edge-derived candidate contours for Images 2 and 5. Images 1, 3 and 4 required thresholding fallback; Image 3 remained poorly segmented.

**4. Why are edges useful for detecting skin lesions?**  
Edges locate brightness changes between a lesion and nearby skin, providing clues to its outer boundary. However, internal pigmentation, hair and skin texture also produce edges. An edge map must therefore be processed into a candidate region before its area can be measured.

**5. What problems did you observe in detecting the lesion boundary?**  
High thresholds lost faint margins. Low thresholds retained internal texture, skin detail and hairs. Image 1 had a prominent crossing hair, Image 2 had low contrast and a jagged contour, Image 3 had many hairs and an unreliable region estimate, and Image 4 had hairs plus a pale outer rim. Morphological closing sometimes joined unrelated edges, while Otsu often omitted faint pigmentation. Image 5 also showed a small unwanted contour extension.

**6. How could your method be improved?**  
Remove hairs before edge detection, correct uneven illumination and tune thresholds per image. Color-based segmentation could help when lesion and skin have similar grayscale values. Better region-selection rules could reject unwanted structures. Expert segmentation masks would allow objective evaluation using Dice or IoU rather than relying only on visual inspection.

## Conclusion
Filtering followed by Canny is useful for exposing lesion structure, but it does not reliably separate every lesion. Gaussian + Canny at 20–40 retained more useful information than the three higher example settings. Morphology gave candidate contours for two images; three required the labelled Otsu fallback, and the basal-cell image remained poorly segmented. The reported measurements describe the extracted candidate regions and are approximate, particularly for the failed example. No clinical accuracy or diagnostic result is claimed.

## Sources and reproducibility
- Assignment requirements: supplied **Lab Assignment.pdf**.
- Dataset: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic . The tested download was dataset version 1.
- Five fixed filenames and all processing parameters are included above. Only the four requested training classes are used; there is no classifier or train/test evaluation in this boundary-detection lab.


## Source URLs and original image files
| Image | Dataset filename | Source URL | Original file | Mask file |
| --- | --- | --- | --- | --- |
| Image 1 | ISIC_0000139.jpg | [Source dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?select=Skin%20cancer%20ISIC%20The%20International%20Skin%20Imaging%20Collaboration%2FTrain%2Fmelanoma%2FISIC_0000139.jpg) | [Original](image_1_original.jpg) | [Mask](image_1_mask.png) |
| Image 2 | ISIC_0024435.jpg | [Source dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?select=Skin%20cancer%20ISIC%20The%20International%20Skin%20Imaging%20Collaboration%2FTrain%2Fpigmented%20benign%20keratosis%2FISIC_0024435.jpg) | [Original](image_2_original.jpg) | [Mask](image_2_mask.png) |
| Image 3 | ISIC_0024504.jpg | [Source dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?select=Skin%20cancer%20ISIC%20The%20International%20Skin%20Imaging%20Collaboration%2FTrain%2Fbasal%20cell%20carcinoma%2FISIC_0024504.jpg) | [Original](image_3_original.jpg) | [Mask](image_3_mask.png) |
| Image 4 | ISIC_0000019.jpg | [Source dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?select=Skin%20cancer%20ISIC%20The%20International%20Skin%20Imaging%20Collaboration%2FTrain%2Fnevus%2FISIC_0000019.jpg) | [Original](image_4_original.jpg) | [Mask](image_4_mask.png) |
| Image 5 | ISIC_0000141.jpg | [Source dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?select=Skin%20cancer%20ISIC%20The%20International%20Skin%20Imaging%20Collaboration%2FTrain%2Fmelanoma%2FISIC_0000141.jpg) | [Original](image_5_original.jpg) | [Mask](image_5_mask.png) |

## Downloadable tables
- [Lesion measurements](lesion_results.csv)
- [Method comparison](method_comparison.csv)

## File manifest
| Filename | Size in bytes |
| --- | --- |
| image_1_mask.png | 1758 |
| image_1_methods.png | 455165 |
| image_1_original.jpg | 939158 |
| image_1_pipeline.png | 515876 |
| image_1_thresholds.png | 308431 |
| image_2_mask.png | 2296 |
| image_2_methods.png | 814413 |
| image_2_original.jpg | 301769 |
| image_2_pipeline.png | 528144 |
| image_2_thresholds.png | 303190 |
| image_3_mask.png | 2153 |
| image_3_methods.png | 849308 |
| image_3_original.jpg | 322215 |
| image_3_pipeline.png | 648660 |
| image_3_thresholds.png | 431014 |
| image_4_mask.png | 1839 |
| image_4_methods.png | 640174 |
| image_4_original.jpg | 107635 |
| image_4_pipeline.png | 580649 |
| image_4_thresholds.png | 363193 |
| image_5_mask.png | 1912 |
| image_5_methods.png | 418213 |
| image_5_original.jpg | 1233699 |
| image_5_pipeline.png | 447368 |
| image_5_thresholds.png | 254335 |
| lesion_results.csv | 588 |
| method_comparison.csv | 1439 |

## Uploading this report
All tables, explanations and source URLs are stored in this Markdown file. Figure links are relative to this file. Keep the images and CSV files beside output.md when uploading, or upload the generated ZIP and extract its contents together. These local figure links are not public hosted URLs.
