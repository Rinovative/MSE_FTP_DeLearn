## Project 01 — CNN on iCoSimal V3

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rinovative/MSE_FTP_DeLearn/blob/main/notebooks/01_cnn_icosimal.ipynb)

---

## Project 02 — Computational graphs

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rinovative/MSE_FTP_DeLearn/blob/main/notebooks/02_cg-linear-regression-stud.ipynb)

---

**Recommended:** Run in Google Colab.

- No local setup required
- All dependencies are installed automatically
- The dataset is downloaded automatically inside the notebook

---

## Local setup (optional)

1. Clone the repository
2. Open it in VS Code
3. Run:

```bash
uv sync
```

This installs all dependencies and creates the virtual environment.

---

## Dataset handling

The dataset is downloaded automatically.

If the automatic download is not possible, use the following fallback:

**In Google Colab:**
- Place the dataset zip at: `/content/icosimal_img_class_03.zip`

**Locally:**
- Place the dataset zip at: `data/icosimal_img_class_03.zip`

In both cases, the dataset will be unpacked automatically into:

```text
data/
└── icosimal_img_class_03/
    └── data_uniform_224_224_sets/
        ├── train/
        └── validate/
```