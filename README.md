# 🍷 Wine Recommender – System Rekomendacji Win

![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-83.3%25-blue)
![HTML](https://img.shields.io/badge/HTML-16.7%25-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)

**Zaawansowany system rekomendacji win wykorzystujący Machine Learning i Deep Learning.**

Projekt porównuje cztery podejścia do problemu rekomendacji: **RP3β** (grafy), **NCF** (sieci neuronowe), **XGBoost** i **Random Forest**.

---

## 📋 Spis treści

- [Przegląd projektu](#przegląd-projektu)
- [Architektura modeli](#architektura-modeli)
- [Instalacja i setup](#instalacja-i-setup)
- [Jak uruchomić](#jak-uruchomić)
- [Dataset](#dataset)
- [Wyniki modeli](#wyniki-modeli)
- [Struktura projektu](#struktura-projektu)
- [Metryki oceny](#metryki-oceny)

---

## 🎯 Przegląd projektu

Projekt bada problemy **collaborative filtering** – jak na podstawie ocen innych użytkowników przewidzieć, które wina spodobają się nowemu użytkownikowi.

### Kluczowe cechy:

✅ **Cztery różne modele ML** – porównanie podejść grafowych, neuronowych i drzewiastych  
✅ **GPU acceleration** – wsparcie dla PyTorch z CUDA  
✅ **Zaawansowane metryki** – RMSE, MAE, R², Precision@K, Recall@K, NDCG@K  
✅ **Analiza eksploracyjna (EDA)** – wizualizacje cech win i zachowań użytkowników  
✅ **Cold-start handling** – obsługa nowych użytkowników i win  

---

## 🧠 Architektura modeli

### 1. **RP3β** – Random Walk Recommendation
Podejście oparte na grafach użytkownik-wino-użytkownik.

**Intuicja:** Jeśli wykonam losowy spacer 3-krokowy w grafie ocen, gdzie najczęściej wylądę?
- Krok 1: Użytkownik → Wino (które oceniał)
- Krok 2: Wino → Inny użytkownik (który to wino oceniał)
- Krok 3: Inny użytkownik → Jego wino

**Parametr `β`** – penalizuje popularne wina, żeby nie polecać wciąż bestsellerów.

**Wyniki:**
```
Precision@10: 0.7008 | Recall@10: 0.8697 | NDCG@10: 0.9571
```

---

### 2. **NCF** – Neural Collaborative Filtering
Sieć neuronowa ucząca embeddingi użytkowników i win.

**Architektura:**
```
User Embedding (50D)  ──┐
                         ├─→ Concat → FC (128) → BatchNorm → ReLU
Wine Embedding (50D)  ──┤     → Dropout → FC (64) → ReLU → FC (1)
                         │
                      Rating
```

**Cechy:**
- Embeddingi pozwalają modelowi uczyć się nienadzorowanych cech (np. "wino wytrawne")
- Early stopping + ReduceLROnPlateau
- Gradient clipping dla stabilności

**Wyniki:**
```
RMSE: 0.4703 | MAE: 0.3555 | R²: 0.333
Precision@10: 0.7455 | Recall@10: 0.9187 | NDCG@10: 0.9832 ⭐
```

---

### 3. **XGBoost** – Gradient Boosting
Ensemble drzew decyzyjnych, uczony na surowych indeksach użytkownika i wina.

**Wyniki:**
```
RMSE: 0.5208 | MAE: 0.4009 | R²: 0.1822
Precision@10: 0.7282 | Recall@10: 0.8996 | NDCG@10: 0.9770
```

---

### 4. **Random Forest** – Bagging Trees
Bazowa linia – kilkaset niezależnych drzew decyzyjnych.

**Wyniki:**
```
RMSE: 0.5235 | MAE: 0.4059 | R²: 0.1736
Precision@10: 0.7215 | Recall@10: 0.8902 | NDCG@10: 0.9764
```

---

## 📊 Porównanie modeli

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ |
|-------|--------|-------|------|-----------------|-------------|-----------|
| **NCF** | 0.4703 | 0.3555 | 0.333 | 0.7455 | **0.9187** | **0.9832** |
| XGBoost | 0.5208 | 0.4009 | 0.1822 | 0.7282 | 0.8996 | 0.9770 |
| RP3β | — | — | — | 0.7008 | 0.8697 | 0.9571 |
| Random Forest | 0.5235 | 0.4059 | 0.1736 | 0.7215 | 0.8902 | 0.9764 |

**🏆 Zwycięzca: NCF** – najlepsze ranking metryki i najmniejszy RMSE.

---

## ⚙️ Instalacja i setup

### Wymagania
- Python 3.8+
- CUDA 12.6 (opcjonalnie – dla GPU)
- ~5 GB dysku na dataset

### Kroki

1. **Klonuj repozytorium:**
   ```bash
   git clone https://github.com/FIlip2423/wine-recommender.git
   cd wine-recommender
   ```

2. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pobierz dataset:**
   Dataset powinien zostać umieszczony w folderze `data/`:
   ```
   data/
   ├── XWines_Slim_150K_ratings.csv
   └── XWines_Slim_1K_wines.csv
   ```

   > **Źródło:** [XWines Dataset](http://www.xwines.it/research/) – oceny win od rzeczywistych użytkowników

4. **Uruchom notebook:**
   ```bash
   jupyter notebook wine_rec_final.ipynb
   ```

---

## 🚀 Jak uruchomić

### Opcja 1: Jupyter Notebook (rekomendowane)
```bash
jupyter notebook wine_rec_final.ipynb
```

Notebook przewódnikiem Cię przez:
1. Wczytanie i eksplorację danych
2. Mapowanie ID na indeksy
3. Zdefiniowanie metryk oceny
4. Trening każdego modelu
5. Porównanie wyników

### Opcja 2: GPU acceleration
Jeśli masz CUDA:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # Powinno wypisać: cuda
```

---

## 📦 Dataset

**XWines Dataset – 150K ocen win**

| Statystyka | Liczba |
|------------|--------|
| Użytkownicy | 10,357 |
| Wina | 1,000 |
| Oceny | 150,000 |
| Skala ocen | 1–5 gwiazdek |
| Gęstość macierzy | 1.45% (bardzo rzadka!) |

**Struktura danych:**
```csv
UserID,WineID,Rating,Date
1200775,155628,5.0,2012-04-19 20:46:00
1088812,167419,3.0,2012-04-21 19:04:41
```

### Cold-Start Problem
Projektu obsługuje "nowych" użytkowników/wina poprzez:
- Usunięcie ich z testów (w głównym notebooku)
- Możliwość treningowania oddzielnych modeli dla cold-start

---

## 📈 Metryki oceny

### Regresja

- **RMSE** (Root Mean Squared Error)
  - Średni błąd przewidywania oceny na skali 1–5
  - Bardziej karze duże błędy
  - Jednostka: gwiazdki

- **MAE** (Mean Absolute Error)
  - Średni błąd absolutny
  - Mniej wrażliwy na outliers
  - Jednostka: gwiazdki

- **R²** (Coefficient of Determination)
  - % wariancji wyjaśnianej przez model
  - 1.0 = idealny, 0.0 = średnia

### Ranking

- **Precision@K**
  - Ile z TOP-K rekomendacji było "trafnych" (ocena ≥ 4)
  - Ogniskuje się na nie polecaniu złych win

- **Recall@K**
  - Jaki % "dobrych" win znalazł się w TOP-K
  - Ogniskuje się na pokryciu wszystkich dobrych win

- **NDCG@K** (Normalized Discounted Cumulative Gain)
  - Czy trafne wina są wysoko w rankingu?
  - Trafienie na 1. miejscu jest warte więcej niż na 10.
  - Najważniejsza metryka dla systemów rekomendacji

---

## 📁 Struktura projektu

```
wine-recommender/
├── README.md                          # Ten plik
├── requirements.txt                   # Zależności
│
├── wine_rec_final.ipynb              # ⭐ GŁÓWNY NOTEBOOK – START TUTAJ
├── wine_recommender_ncf_content.ipynb # NCF z content-based features
│
├── eda.ipynb                          # Eksploracyjna analiza danych
├── eda.html                           # EDA wyeksportowana do HTML
│
├── data/
│   ├── XWines_Slim_150K_ratings.csv  # Oceny użytkowników
│   └── XWines_Slim_1K_wines.csv      # Metadane win
│
├── models/
│   └── best_ncf.pt                    # Najlepszy checkpoint NCF
│
└── src/
    └── (przyszłe: funkcje pomocnicze)
```

---

## 🔬 Wyniki badań

### Główne wnioski

1. **NCF zdominował konkurencję** – głębokie embeddingi lepiej modelują preferencje niż surowe ID
2. **RP3β jest szybki** – brak fazy treningowej, może pracować w real-time
3. **Random Forest jest bazą** – warto mieć jako porównanie dla złożonych modeli
4. **Metryki rankingowe > RMSE** – dla rekomendacji względna kolejność liczy się bardziej niż dokładna ocena

### Wyzwania

- 🔴 **Cold-start** – nowi użytkownicy/wina wymagają hybrid approach
- 🔴 **Rzadkość danych** – macierz ocen ma 98.5% zer
- 🔴 **Skalowanie** – NCF wymaga GPU dla dużych datasetów
- 🟡 **Interpretability** – embeddingi trudne do wyjaśnienia biznesowo

---

## 🛠️ Tech Stack

```
PyTorch 2.10         – Deep Learning
XGBoost 3.2          – Gradient Boosting
scikit-learn 1.8     – Classical ML
NumPy, Pandas        – Data processing
Matplotlib, Seaborn  – Visualization
SciPy                – Scientific computing
```

---

## 📝 Notowania

### Czemu 80/20 split chronologiczny?
W systemach rekomendacji chcemy przewidywać **przyszłe** zachowania, nie snooping przyszłości. Losowy split byłby nierealistyczny.

### Czemu Multiple Models?
Różne podejścia świetlą różne aspekty problemu:
- **Grafy (RP3β)** – szybkie, interpretowalne, real-time
- **Neuron (NCF)** – mocne, ale wymagają dużo danych
- **Ensemble (XGB, RF)** – bezpieczne, szybkie do trenowania

### Czemu NDCG jest najważniejszy?
Użytkownik kliknę na 1. pozycję w rankingu z 90% prawdopodobieństwa.
NDCG chwyta tę hierarchię – to nie jest liniowe!

---

## 🤝 Contributing

Projekt jest open source. Sugestie:

- [ ] Dodać content-based features (opis wina, region, typ)
- [ ] Hybrid model łączący NCF + treści
- [ ] Inference API (FastAPI)
- [ ] Frontend (Streamlit)
- [ ] A/B testing framework
- [ ] Real-time online learning

---

## 📄 Licencja

MIT License – patrz [LICENSE](LICENSE)

---

## 👤 Autor

**Filip** – Student, ML enthusiast 🤖  
GitHub: [@FIlip2423](https://github.com/FIlip2423)

---

## 📚 Przydatne linki

- [XWines Dataset](http://www.xwines.it/research/)
- [NCF Paper](https://arxiv.org/abs/1708.05024)
- [RP3β Paper](https://arxiv.org/abs/1412.6612)
- [PyTorch Docs](https://pytorch.org/docs/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)

---

**⭐ Jeśli projekt Ci się spodobał, daj gwiazdkę!**

Ostatnia aktualizacja: 2026-10-02
