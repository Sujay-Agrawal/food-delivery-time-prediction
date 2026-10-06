# 🛵 Food Delivery Time Prediction using Regression

Predicting how long a food delivery will take using Linear Regression and its variants (Ridge, Lasso, Polynomial), built with Python and scikit-learn. The project also compares **Label Encoding vs One-Hot Encoding** for handling categorical features.

## 📌 Problem Statement

Delivery time is one of the biggest factors in customer satisfaction for food delivery platforms. This project builds machine learning models that predict **delivery time (in minutes)** from factors such as distance, traffic, weather, time of day, vehicle type, preparation time, and courier experience.

## 📂 Dataset

- **File:** `Food_Delivery_Times.csv`
- **Size:** 1,000 rows × 9 columns
- **Target variable:** `Delivery_Time_min`
- **Missing values:** 30 each in `Weather`, `Traffic_Level`, `Time_of_Day`, and `Courier_Experience_yrs`

| Feature | Type | Description |
|---|---|---|
| `Order_ID` | Numeric | Unique order identifier |
| `Distance_km` | Numeric | Distance between restaurant and customer (avg ≈ 10 km) |
| `Weather` | Categorical | Weather condition (Clear, Foggy, Rainy, Windy, ...) |
| `Traffic_Level` | Categorical | Low / Medium / High |
| `Time_of_Day` | Categorical | Morning, Afternoon, Evening, Night |
| `Vehicle_Type` | Categorical | Bike, Scooter, ... |
| `Preparation_Time_min` | Numeric | Time the restaurant took to prepare the order (avg ≈ 17 min) |
| `Courier_Experience_yrs` | Numeric | Courier's experience in years (avg ≈ 4.6 yrs) |
| `Delivery_Time_min` | Numeric | **Target:** total delivery time |

## 🔧 Project Workflow

1. **Data exploration:** checked shape, columns, summary statistics, and missing values
2. **Data cleaning:**
   - Categorical columns (`Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`) filled with the **mode**
   - `Courier_Experience_yrs` filled with the **median**
3. **Visualization:** correlation heatmap to understand relationships between variables
4. **Encoding (compared two approaches):**
   - **Label Encoding** (`LabelEncoder`)
   - **One-Hot Encoding** (`pd.get_dummies`)
5. **Train/test split:** 80% training / 20% testing (`random_state=42`)
6. **Modeling:**
   - Linear Regression (with each encoding)
   - Ridge Regression
   - Lasso Regression
   - Polynomial Regression (degree 2)
7. **Evaluation:** R² score, Adjusted R², and train vs test R² to check for overfitting

## 📊 Results

**Encoding comparison (Linear Regression):**

| Encoding | R² | Adjusted R² |
|---|---|---|
| Label Encoding | 0.7563 | 0.7541 |
| One-Hot Encoding | **0.8185** | **0.8148** |

**Model comparison (One-Hot Encoded data, test set):**

| Model | R² Score |
|---|---|
| Linear Regression | **0.8185** |
| Ridge Regression | 0.8183 |
| Lasso Regression | 0.7600 |
| Polynomial Regression (degree 2) | 0.7637 |

**Overfitting check (Linear Regression, One-Hot):** Train R² = 0.7608 vs Test R² = 0.8185. Train and test scores are close, so there are no signs of serious overfitting.

> **Best model:** Linear Regression with One-Hot Encoding (R² ≈ 0.82). Ridge performs almost identically.

## 💡 Key Learnings

- **One-Hot Encoding beat Label Encoding** by about 6 points of R² (0.756 → 0.818), because label encoding implies a false order between categories like weather types
- Ridge and Linear Regression performed almost the same, so regularization didn't add much on this small, low-noise dataset
- Lasso scored lower (0.760), likely because it shrinks some feature coefficients too aggressively with default settings
- A more complex model (Polynomial) did not beat the simple one, so a simpler model can be the better choice
- Adjusted R² and train-vs-test comparison help confirm a model isn't just memorizing the data

## ⚠️ Limitations & Future Work

- Small dataset (1,000 rows), so results may not generalize
- Only R² and Adjusted R² are reported; adding **MAE and RMSE** would show the error in actual minutes
- Hyperparameters of Ridge/Lasso were left at defaults; `GridSearchCV` could tune `alpha`
- Try tree-based models (Random Forest, Gradient Boosting) and cross-validation
- Remove the non-informative `Order_ID` column from the features
- Keep decimal values in `Distance_km` and `Courier_Experience_yrs` when converting the one-hot data to numbers
- Fit `PolynomialFeatures` on training data only, and use `transform` on the test data

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Jupyter Notebook`

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/food-delivery-time-prediction.git
cd food-delivery-time-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Open the notebook
jupyter notebook Regression_models.ipynb
```

> Make sure `Food_Delivery_Times.csv` is in the same folder as the notebook. If your file is named `Food_Delivery_Times (1).csv`, rename it or update the filename in the notebook.

## 📬 Connect

Built while learning machine learning. Feedback and suggestions are welcome!

- LinkedIn: [Sujay Agrawal](https://www.linkedin.com/in/sujay-agrawal-3aa09a30a/)
- GitHub: [Sujay-Agrawal](https://github.com/Sujay-Agrawal)
