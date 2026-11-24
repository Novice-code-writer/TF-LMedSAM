# Statistical tables

Statistical analysis: Dice and IoU estimates are reported with 95% percentile bootstrap confidence intervals from 2,000 image resamples. Paired Dice differences use the same image and reference-class eligibility across methods. Two-sided sign-flip permutation tests use 4,999 random assignments, an add-one correction, and Holm adjustment within each comparison family. All lesion classes within an image remain together. Resamples missing an originally represented class are excluded and counted. Empty reference classes have undefined Dice and IoU; their false-positive counts are retained separately. Entirely reference-empty images are excluded from Dice distributions. The intervals characterize test-image variation conditional on fixed outputs.

## Table 2

| Condition | Dataset / direction | Dice (%) | 95% CI (%) | IoU (%) | 95% CI (%) |
| --- | --- | ---: | --- | ---: | --- |
| U-Net | IDRiD | 65.92 | 62.09–69.43 | 52.10 | 47.89–56.14 |
| U-Net | DDR | 58.20 | 56.66–59.84 | 45.30 | 43.73–46.96 |
| Att-UNet | IDRiD | 68.45 | 65.76–70.93 | 54.20 | 51.34–57.02 |
| Att-UNet | DDR | 60.10 | 58.61–61.56 | 47.10 | 45.54–48.62 |
| CE-Net | IDRiD | 70.12 | 67.54–72.70 | 56.30 | 53.36–59.24 |
| CE-Net | DDR | 62.30 | 60.90–63.69 | 49.20 | 47.71–50.69 |
| TransUNet | IDRiD | 72.50 | 69.97–75.05 | 59.10 | 56.04–62.18 |
| TransUNet | DDR | 65.40 | 64.05–66.77 | 52.40 | 50.99–53.90 |
| Swin-Unet | IDRiD | 74.32 | 71.66–76.97 | 61.20 | 57.93–64.47 |
| Swin-Unet | DDR | 67.80 | 66.55–69.03 | 54.60 | 53.20–55.98 |
| MedSAM (Full) | IDRiD | 76.12 | 73.35–78.69 | 63.50 | 60.09–66.75 |
| MedSAM (Full) | DDR | 69.50 | 68.30–70.70 | 56.80 | 55.45–58.13 |
| MedSAM Adapter | IDRiD | 76.85 | 74.46–79.00 | 64.10 | 61.20–66.82 |
| MedSAM Adapter | DDR | 70.20 | 68.93–71.51 | 57.50 | 56.11–58.99 |
| TF-LMedSAM | IDRiD | 78.42 | 76.58–80.12 | 66.80 | 64.65–68.85 |
| TF-LMedSAM | DDR | 72.15 | 71.42–72.93 | 59.90 | 59.11–60.75 |

## Table 3

| Condition | Dataset / direction | Dice (%) | 95% CI (%) | IoU (%) | 95% CI (%) |
| --- | --- | ---: | --- | ---: | --- |
| Baseline (Frozen MedSAM) | IDRiD | 59.00 | 54.74–63.44 | 45.20 | 40.81–49.81 |
| + Visual Adapter | IDRiD | 74.60 | 72.46–76.62 | 60.50 | 57.91–62.95 |
| + TPFM | IDRiD | 77.80 | 76.58–78.98 | 64.10 | 62.48–65.67 |
| + Uncertainty Loss (Full) | IDRiD | 78.42 | 76.58–80.12 | 66.80 | 64.65–68.85 |

## Table 4

| Condition | Dataset / direction | Dice (%) | 95% CI (%) | IoU (%) | 95% CI (%) |
| --- | --- | ---: | --- | ---: | --- |
| U-Net | cross_IDRiD_to_DDR | 52.30 | 51.82–52.80 | — | — |
| U-Net | cross_DDR_to_IDRiD | 55.10 | 53.82–56.52 | — | — |
| TransUNet | cross_IDRiD_to_DDR | 58.60 | 58.13–59.07 | — | — |
| TransUNet | cross_DDR_to_IDRiD | 61.20 | 59.91–62.38 | — | — |
| MedSAM | cross_IDRiD_to_DDR | 65.40 | 65.02–65.78 | — | — |
| MedSAM | cross_DDR_to_IDRiD | 68.70 | 67.40–70.14 | — | — |
| TF-LMedSAM | cross_IDRiD_to_DDR | 70.50 | 70.25–70.75 | — | — |
| TF-LMedSAM | cross_DDR_to_IDRiD | 73.20 | 72.50–73.86 | — | — |

## Paired Dice differences

| Comparison | Difference (pp) | 95% CI (pp) | Raw p | Holm p |
| --- | ---: | --- | ---: | ---: |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/U-Net | 12.50 | 8.57–16.45 | 0.0002 | 0.0014 |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/Att-UNet | 9.97 | 6.66–13.23 | 0.0002 | 0.0014 |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/CE-Net | 8.30 | 5.66–10.93 | 0.0002 | 0.0014 |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/TransUNet | 5.92 | 3.39–8.63 | 0.0004 | 0.0016 |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/Swin-Unet | 4.10 | 1.08–7.18 | 0.0130 | 0.0390 |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/MedSAM (Full) | 2.30 | -0.73–5.06 | 0.1296 | 0.2592 |
| comparison/IDRiD/TF-LMedSAM vs comparison/IDRiD/MedSAM Adapter | 1.57 | -1.32–4.37 | 0.2946 | 0.2946 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/U-Net | 13.95 | 12.13–15.71 | 0.0002 | 0.0014 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/Att-UNet | 12.05 | 10.45–13.58 | 0.0002 | 0.0014 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/CE-Net | 9.85 | 8.06–11.59 | 0.0002 | 0.0014 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/TransUNet | 6.75 | 5.09–8.36 | 0.0002 | 0.0014 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/Swin-Unet | 4.35 | 3.00–5.73 | 0.0002 | 0.0014 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/MedSAM (Full) | 2.65 | 1.22–4.12 | 0.0012 | 0.0024 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/MedSAM Adapter | 1.95 | 0.41–3.35 | 0.0110 | 0.0110 |
| ablation/IDRiD/+ Visual Adapter vs ablation/IDRiD/Baseline (Frozen MedSAM) | 15.60 | 10.43–20.70 | 0.0002 | 0.0006 |
| ablation/IDRiD/+ TPFM vs ablation/IDRiD/+ Visual Adapter | 3.20 | 0.89–5.60 | 0.0140 | 0.0280 |
| comparison/IDRiD/TF-LMedSAM vs ablation/IDRiD/+ TPFM | 0.62 | -1.69–2.70 | 0.5926 | 0.5926 |
| cross_IDRiD_to_DDR/DDR/TF-LMedSAM vs cross_IDRiD_to_DDR/DDR/U-Net | 18.20 | 17.63–18.75 | 0.0002 | 0.0006 |
| cross_IDRiD_to_DDR/DDR/TF-LMedSAM vs cross_IDRiD_to_DDR/DDR/TransUNet | 11.90 | 11.36–12.44 | 0.0002 | 0.0006 |
| cross_IDRiD_to_DDR/DDR/TF-LMedSAM vs cross_IDRiD_to_DDR/DDR/MedSAM | 5.10 | 4.63–5.62 | 0.0002 | 0.0006 |
| cross_DDR_to_IDRiD/IDRiD/TF-LMedSAM vs cross_DDR_to_IDRiD/IDRiD/U-Net | 18.10 | 16.71–19.47 | 0.0002 | 0.0006 |
| cross_DDR_to_IDRiD/IDRiD/TF-LMedSAM vs cross_DDR_to_IDRiD/IDRiD/TransUNet | 12.00 | 10.50–13.54 | 0.0002 | 0.0006 |
| cross_DDR_to_IDRiD/IDRiD/TF-LMedSAM vs cross_DDR_to_IDRiD/IDRiD/MedSAM | 4.50 | 3.13–5.83 | 0.0002 | 0.0006 |
| training_fraction/DDR/TF-LMedSAM/10 vs training_fraction/DDR/TransUNet/10 | 25.54 | 23.80–27.06 | 0.0002 | 0.0012 |
| training_fraction/DDR/TF-LMedSAM/20 vs training_fraction/DDR/TransUNet/20 | 19.74 | 18.00–21.48 | 0.0002 | 0.0012 |
| training_fraction/DDR/TF-LMedSAM/40 vs training_fraction/DDR/TransUNet/40 | 13.34 | 11.72–15.05 | 0.0002 | 0.0012 |
| training_fraction/DDR/TF-LMedSAM/60 vs training_fraction/DDR/TransUNet/60 | 8.60 | 6.98–10.27 | 0.0002 | 0.0012 |
| training_fraction/DDR/TF-LMedSAM/80 vs training_fraction/DDR/TransUNet/80 | 7.27 | 5.64–8.89 | 0.0002 | 0.0012 |
| comparison/DDR/TF-LMedSAM vs comparison/DDR/TransUNet | 6.75 | 5.09–8.36 | 0.0002 | 0.0012 |
