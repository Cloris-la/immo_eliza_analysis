# immo_eliza_analysis# 🏠 ImmoEliza - Real Estate Data Analysis

## 📌 Description

This project is the continuation of a data scraping mission for the company **ImmoEliza**. The objective here is to perform an **exploratory data analysis (EDA)** on a real estate dataset collected from Belgian listings, in order to extract actionable insights that will later support a machine learning model for price prediction.

The dataset contains over **85,000 properties** and includes key features like price, location, type, surface, and amenities such as gardens, terraces, and pools.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/immo_eliza_analysis.git
cd immo_eliza_analysis
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
# on Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

To launch the analysis, open the notebook:

```
jupyter notebook
```

Then open analysis.ipynb or your analysis script, and execute the cells to:

- Clean the dataset

- Explore correlations

- Generate visualizations

- Interpret results and draw conclusions

Make sure your `immoweb-dataset.csv` file is available in the working directory or update the file path accordingly.

### 📈 Analysis Objectives

- Clean the data (remove duplicates, fix formatting, handle missing values)

- Visualize price distributions and regional differences

- Measure correlation between features (e.g., price vs surface)

- Identify outliers and influential variables

- Answer key business questions, including:

  - What are the most/least expensive municipalities?

  - Which features influence price the most?

  - Where are the highest price per m² in Belgium?

Visualizations include bar plots, histograms, and heatmaps.

### 🧠 Tech Stack

- Python

- pandas

- matplotlib

- seaborn

- Jupyter Notebook
