# Impact of Evolutionary Black-Box Adversarial Attacks on Machine Learning Accuracy and Predictive Outcomes

Code accompanying the manuscript submitted to *Scientific Reports* by Saira Akram, Sibghat Ullah, Qazi Mudassar Ilyas, Shafi Ullah, Sajid Iqbal, Muhammad Nabeel Asghar, Ahmed Abdullatif Alyahya and Muhammad Aasim Rafique.

The repository contains the four Colab notebooks used to produce the reported results: a lightweight CNN traffic sign classifier trained on GTSRB with a geometric data augmentation robustness check, an evolutionary (genetic algorithm) black-box targeted attack, a saliency-map analysis, and a Boundary Attack comparison using the Adversarial Robustness Toolbox (ART).

## Notebooks

| # | Notebook | Where it was run | Content |
|---|----------|------------------|---------|
| 1 | `1_Model_Building_and_Robustness_Check_by_Data_Augmentationain_file.ipynb` | Local machine (Python 3.9) | Baseline CNN training and evaluation (accuracy, F1, confusion matrix), geometric data augmentation robustness check |
| 2 | `2_Quantitative_Evaluation_of_Source_Target_Adversarial_Attacks_Across_Multiple_Samples_and_Class_Pairs.ipynb` | Google Colab (GPU) | Genetic algorithm attack over source-target class pairs, 5 images per pair; reports attack success rate, mean and standard deviation of the L2 distance |
| 3 | `3_Saliency_Map_of_Stop_Sign.ipynb` | Google Colab | Saliency map analysis of an original and an adversarial stop sign image |
| 4 | `4_Boundary_Attack_using_ART.ipynb` | Google Colab (GPU) | Boundary Attack on a single image using ART |

Each notebook is self-contained and trains or loads on the same model.

### Which results are in the paper

The saved cell outputs in the notebooks are the results reported in the paper. In notebook 2, the cells marked `REPORTED IN PAPER` are the seven source-target pairs reported in the manuscript (35 adversarial tests). Notebook 2 also contains further pair runs (18 pair runs in total; the pair 17 to 14 was run twice) that were kept for completeness and are not reported in the manuscript.

## Dataset

German Traffic Sign Recognition Benchmark (GTSRB):

> J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition." *Neural Networks*, 32, 323-332, 2012. https://doi.org/10.1016/j.neunet.2012.02.016

Source used here: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign (official site: https://benchmark.ini.rub.de).

The dataset is not redistributed in this repository.

- **Notebook 1** expects the downloaded dataset in a local folder `dataset22` with `Train/` (43 class sub-folders), `Test/`, `Meta/`, `Train.csv`, `Test.csv` and `Meta.csv`. The paths in the notebook are the ones used on the original machine; replace them with the path of your own copy (the cells say `enter the path`).
- **Notebooks 2, 3, 4** download the dataset in Google Colab through the Kaggle API. You need your own Kaggle API token (`kaggle.json`) when the notebook asks you to upload it. Never commit this file to a repository.

No custom dataset is released. Adversarial images are generated when the notebooks are run and are shown in the cell outputs (they are not saved to disk).

## Data splitting and pre-processing

| | Notebook 1 | Notebooks 2, 3, 4 |
|---|---|---|
| Data | Official training set (39,209 images) | Official training folders |
| Resize | 25×25×3 | 25×25×3 |
| Scaling | None (pixel values 0-255) | Divided by 255 (0-1) |
| Split | `train_test_split`, 80/20, `random_state=0`: 31,367 training / 7,842 validation | Keras `ImageDataGenerator(validation_split=0.2)`: 31,368 training / 7,841 validation |
| Final test | Official test set (12,630 images, `Test.csv`) | Not evaluated in these notebooks |

## Models

- **Notebook 1:** four Conv2D layers (32, 32, 64, 64 filters) with max pooling and dropout (0.25, 0.25, 0.5), a 256-unit dense layer and a 43-way softmax; Adam, categorical cross-entropy, batch size 32, 40 epochs. 160,331 parameters. Saved to `training/TSR.keras` (created when notebook 1 is run). Baseline accuracy on the official test set (12,630 images): **94.25%**.
- **Notebook 2 (attack model):** same model as notebook 1, trained on inputs scaled to 0-1 for 40 epochs with the seed below, and saved to `/content/model.h5` in the Colab session.
- **Notebook 3:** the same architecture trained for 20 epochs on inputs scaled to 0-1.
- **Notebook 4:** a smaller CNN (two Conv2D layers, a 128-unit dense layer) trained for 40 epochs, wrapped with ART.

## Genetic algorithm (notebook 2)

The attack implemented in notebook 2 is a targeted black-box genetic algorithm that only uses the model's output probabilities:

- Population of 100 candidate perturbations, initialised uniformly in [-0.1, 0.1] per pixel; 100 generations (at most 10,000 model queries per image).
- Fitness: predicted probability of the target class for the clipped image (values in [0, 1]).
- Selection: tournament of two; variation: with probability 0.5 a child receives Gaussian noise with standard deviation 0.02.
- No early stopping: all 100 generations are run.
- An attack is successful when the arg-max prediction for the final adversarial image equals the target class. The L2 distance is computed between the clean and the adversarial image.
- Source images are drawn randomly from the GTSRB training folders (5 per source-target pair).

## Reproducibility

- Notebook 2 sets `SEED = 42` (`random`, NumPy and TensorFlow). Because the model is trained in the notebook on a GPU, re-running it will not give identical numbers; the results reported in the paper are the saved outputs.
- Notebook 1 does not set a global seed for model training (only `random_state=0` for the data split; `np.random.seed(42)` appears only in the plotting cell after training).
- Notebook 1 hardware and software (as reported in the paper): Intel Core i7-10700, 32 GB RAM, NVIDIA GTX 1650, Python 3.9 with TensorFlow 2.10. Notebooks 2 to 4 ran on Google Colab GPU runtimes.

## Limitations

As discussed in the paper, the quantitative evaluation uses 5 images per source-target class pair, a lightweight CNN with 25×25 inputs, no explicit perturbation budget, a single dataset and a single seed.

## Citation

```
Akram S., Ullah S., Ilyas Q. M., Ullah S., Iqbal S., Asghar M. N., Alyahya A. A., Rafique M. A.
Impact of Evolutionary Black-Box Adversarial Attacks on Machine Learning Accuracy and Predictive Outcomes.
Scientific Reports, [year, DOI once available].
```

Archived version of this code: [Zenodo DOI].

## License

MIT License (see `LICENSE`).

## Contact

Corresponding author: Muhammad Aasim Rafique, King Faisal University (mrafique@kfu.edu.sa).
