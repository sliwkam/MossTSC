# 🧊 Matrix of Series-to-Space Symbols (MOSS) – CNN for Time Series Classification

This repository contains the implementation of **MOSS (Matrix of Series-to-Space Symbols)**, a novel **Convolutional Neural Network (CNN)** architecture designed for **time series classification**. The method transforms time series data into a structured spatial representation, enabling improved classification performance.

📄 **Reference:**  
🔗 *[Paper Title](#) – Author Name et al., 2025*  


## 📌 Features
- 📊 **Series-to-Space transformation** – converts time series into structured spatial representations.
- 🧠 **CNN-based classification** – uses convolutional layers for feature extraction.
- 🔬 **Extensive benchmarking** – tested on multiple datasets with superior results.


## 📂 Repository structure

```plaintext
project-root/
├── src/                                    # Source code files
│   ├── _0_run_experiment.py                # Main script to run experiments
│   ├── _1_DataProcessor_.py                # Data loading and processing logic
│   ├── _2_discretization.py                # Data discretization utilities
│   ├── _6_reshape_array.py                 # Reshaping arrays
│   ├── _6a_statistical_reduction.py        # Statistical reduction methods
│   ├── _6b_matrix_reduction.py             # Matrix dimensionality reduction
│   ├── _7_standarization.py                # Data standardization functions
│   ├── _8_data_transformation.py           # Various data transformations
│   ├── _9_models_architecture.py           # Model architectures
│   ├── _9a_models_run.py                   # Training/testing scripts
│   ├── _99_pipeline_utils_.py              # Pipeline utility scripts
│   ├── __init__.py                         # Makes `src` a Python package
│   ├── discretization_module_utils.py      # Utilities for data discretization
│   ├── experiment_utils.py                 # Helper functions for experiments
│   └── json_to_csv.py                      # Convert JSON files to CSV
├── notebooks/                              # Jupyter notebooks or project demos
│   └── example_run.ipynb                   # Example usage / demonstration
├── results/                                # Output or result data
│   ├── dataset_estimators_accuracy.csv     # Avg accuracy per dataset (MOSS & selected classifiers)
│   └── seed_accuracy.csv                   # Accuracy of MOSS classifier per seed for each dataset
├── .gitignore                              # Files/directories to ignore in Git
├── CITATION.cff                            # Citation file 
├── LICENSE                                 # Project license
├── requirements.txt                        # Python dependencies
└── README.md                               # Project overview and documentation
```


## ⚙️ Installation & dependencies
Ensure you have **Python 3.8+** installed, then run:

```bash
git clone https://github.com/sliwkam/MossTSC.git
cd MossTSC
pip install -r requirements.txt
```

To set up a Conda environment:
```bash
conda create --name moss-env python=3.9
conda activate moss-env
pip install -r requirements.txt
```

## 🚀 Usage
To run the model, open and execute the Jupyter Notebook provided in the repository:
```bash
notebooks/example_run.ipynb
```
This notebook contains step-by-step instructions on how to load, configure, and use the model. Make sure you have all dependencies installed before running it.


## 📊 Results
* Achieves competitive accuracy on benchmark time series datasets, performing on par with state-of-the-art neural network classifiers.
* Demonstrates improved generalization compared to traditional CNN-method.
* Detailed results and visualizations are available in the /results folder.

## 📜 Citation
If you use MOSS in your research, please cite our paper:

```bibtex
@article{MOSSTSC2025,
  author = {},
  title = {},
  journal = {Journal Name},
  year = {2025},
  doi = {10.XXXX/XXXXXXX}
}
```

## 📌 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 📩 Contact
For questions or collaborations, feel free to reach out:

📧 Email: mateusz.sliwka.dokt@pw.edu.pl
🌐 Website: [LinkedIn](https://www.linkedin.com/in/mateusz-sliwka/)

🚀 Happy coding & research! 🎯
