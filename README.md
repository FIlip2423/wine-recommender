# 🍷 Wine Recommender System

Advanced wine recommendation system using machine learning and deep learning techniques. This project compares multiple approaches to the recommendation problem and identifies which performs best for predicting user preferences.

## What does this project do?

The goal is to predict which wines a specific user is likely to enjoy based on their rating history. The model learns from rating data and predicts which wines are most likely to receive high scores from a given user.

The project compares four different approaches:
- **RP3β** – graph-based recommendations using user-item walks
- **NCF** – Neural Collaborative Filtering with embeddings
- **XGBoost** – gradient boosting model
- **Random Forest** – classical tree ensemble model

## Key Results

| Model | RMSE | MAE | R² | Precision@10 | Recall@10 | NDCG@10 |
|-------|------|-----|----|--------------|----------|---------|
| **NCF** | 0.4703 | 0.3555 | 0.333 | 0.7455 | **0.9187** | **0.9832** |
| XGBoost | 0.5208 | 0.4009 | 0.1822 | 0.7282 | 0.8996 | 0.9770 |
| RP3β | — | — | — | 0.7008 | 0.8697 | 0.9571 |
| Random Forest | 0.5235 | 0.4059 | 0.1736 | 0.7215 | 0.8902 | 0.9764 |

**NCF achieves the best overall performance** with superior ranking metrics and lowest prediction error (RMSE).

## Dataset

The project uses the XWines Slim dataset:
- 150,000 ratings
- 10,561 users
- 1,007 wines
- Time period: 2012-04-19 to 2021-12-31
- Average rating: ~3.82 / 5 stars
- Highly sparse data (typical for recommendation systems)

## Repository Structure

```
wine-recommender/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── eda.ipynb                     # Main exploratory data analysis
├── eda.html                      # EDA exported to HTML
├── wine_rec_final.ipynb         # Final model comparison notebook
├── wine_recommender_ncf_content.ipynb
├── model_comparison.png
├── archive/                      # Experimental and duplicate notebooks
│   └── README.md
└── data/                         # Dataset (local environment)
```

**Note:** Experimental and duplicate notebooks are stored in `archive/` to keep the main repository clean, organized, and professionally presented on GitHub.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/FIlip2423/wine-recommender.git
cd wine-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter:
```bash
jupyter notebook
```

4. Open notebooks:
- `eda.ipynb` – Data analysis and exploration
- `wine_rec_final.ipynb` – Complete model training and evaluation pipeline

## Data Analysis

The `eda.ipynb` notebook contains key analyses:
- Rating distribution and statistics
- User and wine counts
- Cold-start problem analysis
- Wine popularity patterns
- Temporal trends
- Long-tail distribution analysis

## Evaluation Metrics

The project uses both regression and ranking metrics:

**Regression Metrics:**
- **RMSE** – Root Mean Squared Error (rating scale)
- **MAE** – Mean Absolute Error
- **R²** – Coefficient of Determination

**Ranking Metrics:**
- **Precision@K** – Proportion of relevant items in top-K
- **Recall@K** – Coverage of relevant items
- **NDCG@K** – Ranking quality with position weighting

For recommendation systems, **ranking metrics are more important than exact rating prediction**.

## Model Architectures

### RP3β – Graph-Based Approach
Random walk on user-item-user graph. Fast, interpretable, real-time capable.

### NCF – Neural Collaborative Filtering
Deep learning with user/item embeddings. Best overall performance but requires GPU for large datasets.

### XGBoost & Random Forest
Classical ensemble methods. Fast training, competitive results, good baselines.

## Technologies

- **PyTorch** – Deep learning and GPU support
- **XGBoost** – Gradient boosting
- **scikit-learn** – Machine learning algorithms
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing
- **Matplotlib/Seaborn** – Visualization
- **SciPy** – Scientific computing

## Main Findings

✅ **NCF dominates** – Deep embeddings better capture user preferences than raw IDs  
✅ **RP3β is fast** – No training phase, suitable for real-time systems  
✅ **Random Forest serves as baseline** – Good comparison for complex models  
✅ **Ranking metrics > RMSE** – Relative ordering matters more than exact predictions  

## Challenges

🔴 **Cold-start problem** – New users/items lack history  
🔴 **Data sparsity** – Rating matrix is 98.5% zeros  
🔴 **Scalability** – NCF needs GPU for large datasets  
🟡 **Interpretability** – Embeddings are difficult to explain

## Archive

Experimental notebooks and duplicates are preserved in the `archive/` directory:
- Maintains project history
- Keeps main repo clean and focused
- Supports professional GitHub presentation
- All exploratory work remains available as reference

## Author

**Filip** – ML Engineer & Data Science Enthusiast  
GitHub: [@FIlip2423](https://github.com/FIlip2423)

## License

MIT License – Free for educational and experimental use.

---

**⭐ If you find this project useful, please consider starring it!**

Last updated: 2026-10-02
