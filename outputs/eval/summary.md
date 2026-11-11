| group | widest_gap | ground_fraction | enemies_per_100 | qblocks_per_100 | pipes_valid | copy_rate | pipe_errors |
|---|---|---|---|---|---|---|---|
| real (all 15) | 14.93 ± 16.38 | 0.76 ± 0.25 | 8.92 ± 5.49 | 3.51 ± 3.56 | 100% | 0.89 ± 0.29 | 0.00 ± 0.00 |
| real train (13) | 16.15 ± 17.28 | 0.75 ± 0.27 | 9.18 ± 5.23 | 3.44 ± 3.71 | 100% | 1.00 ± 0.00 | 0.00 ± 0.00 |
| real val ['mario-1-2', 'mario-6-1'] | 7.00 ± 0.00 | 0.85 ± 0.05 | 7.23 ± 6.69 | 3.98 ± 2.35 | 100% | 0.14 ± 0.10 | 0.00 ± 0.00 |
| T=0.5 raw | 5.46 ± 11.84 | 0.97 ± 0.07 | 1.07 ± 0.76 | 0.15 ± 0.91 | 88% | 0.93 ± 0.13 | 0.84 ± 3.17 |
| T=0.5 +ground fix | 1.54 ± 1.43 | 0.99 ± 0.01 | 1.07 ± 0.76 | 0.15 ± 0.91 | 88% | 0.93 ± 0.13 | 0.84 ± 3.17 |
| T=1.0 raw | 7.36 ± 5.87 | 0.85 ± 0.12 | 10.81 ± 2.33 | 3.50 ± 2.08 | 0% | 0.17 ± 0.08 | 27.88 ± 17.68 |
| T=1.0 +ground fix | 3.18 ± 0.77 | 0.91 ± 0.05 | 10.81 ± 2.33 | 3.50 ± 2.08 | 0% | 0.17 ± 0.07 | 27.88 ± 17.68 |
| T=1.5 raw | 11.42 ± 4.17 | 0.57 ± 0.13 | 15.20 ± 3.46 | 8.41 ± 2.87 | 0% | 0.00 ± 0.00 | 52.36 ± 15.63 |
| T=1.5 +ground fix | 3.70 ± 0.54 | 0.75 ± 0.07 | 15.20 ± 3.46 | 8.41 ± 2.87 | 0% | 0.00 ± 0.00 | 52.36 ± 15.63 |

Generated levels: 50 per temperature, 200 columns each, top-k=0.
Real levels are 149-373 columns wide, so 'widest gap' is not perfectly comparable (longer levels have more chances for a wide gap).
Ground fix: gaps wider than 4 filled; mean gaps filled per level: T=0.5: 0.24, T=1.0: 1.50, T=1.5: 4.78
Copy-rate reference: the 2 held-out real levels have 0.14 of their 8-column windows in the training levels.
