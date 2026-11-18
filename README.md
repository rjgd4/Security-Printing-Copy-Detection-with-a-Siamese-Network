# Security Printing: Copy Detection with a Siamese Network

Can a neural network tell an original printed anti-copy code from a copy? MATLAB project at Institut d'Optique Graduate School (2026), done in pairs.

## Idea

Anti-copy codes are small random black-and-white images printed on a product. Copying one means scanning and reprinting it, and each print-and-scan step degrades the image a little (ink spreading, flipped dots, noise). The goal is to detect that difference automatically.

## What it does

1. **Generates the data**: 200 training codes and 100 test codes. For each code, two originals (`p`, `q`) and one copy (`r`).
2. **Simulates the forger**: the copy is made from the scan `p` only. The code is estimated (2×2 averaging + Otsu threshold) and reprinted with `printScan`. About 4.5 % of the code dots are wrongly recovered.
3. **Trains a Siamese network**: two identical convolutional branches produce a 4096-value fingerprint each. The absolute difference goes through a fully connected layer and a sigmoid to give a similarity score between 0 and 1.
4. **Evaluates** on 1000 test pairs, with a decision threshold calibrated on the training data.

## Results

| Measure | Value |
|---|---|
| Training | 200 iterations, 10 pairs each (about 90 s) |
| Decision threshold (from training data) | 0.80 |
| Test accuracy (1000 pairs) | **100 %** |
| AUC | **1.000** |

Similar pairs (two originals) score between 0.80 and 0.88, and original/copy pairs score below 0.1, with no overlap.

## Next steps

- Other ink-spreading levels (`dotG`) for harder copies
- Longer training
- A stronger forger model

## Run it

Requires MATLAB (or MATLAB Online) with the Deep Learning, Image Processing and Statistics and Machine Learning toolboxes.

1. Open MATLAB in the `code/` folder
2. Run `LANCER_PROJET.m` (or type `security_printing_project`)
3. The data is generated automatically and all figures appear after 1 to 2 minutes

## Files

- `code/security_printing_project.m`: full pipeline (data, network, training, evaluation)
- `code/LANCER_PROJET.m`: launcher
- `rapport/`: report (in French)

## Credits

The Siamese network structure and the `printScan` function come from the course material (Security Printing, Institut d'Optique). The pair generation, the forger, the training and the evaluation are our work.

## Authors

Robin Jegard, Gaspard Milek
