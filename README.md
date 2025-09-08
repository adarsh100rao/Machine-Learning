🏀 NBA 2K Player Data Preprocessing

This project processes and transforms the NBA 2K dataset to prepare it for machine learning tasks such as salary prediction or player performance analysis.

The script handles:

Data cleaning (height, weight, salary, country, draft info, etc.)

Feature engineering (age, experience, BMI, etc.)

Multicollinearity removal

Data transformation (scaling numeric features & one-hot encoding categorical features)

📂 Dataset

Source: nba2k-full.csv

The script automatically downloads the dataset if it’s not present in the ../Data/ folder.

⚙️ Features Implemented
🔹 Data Cleaning

Convert birthdays & draft years to datetime.

Handle missing team values (No Team).

Convert height (feet/meters) → meters.

Convert weight (lbs/kg) → kilograms.

Remove $ from salaries.

Categorize country (USA vs Not-USA).

Handle undrafted players (draft_round).

🔹 Feature Engineering

age → based on version year and birthday.

experience → based on version year and draft year.

BMI → weight / height².

Drop high-cardinality categorical features (>50 unique values).

🔹 Multicollinearity

Remove highly correlated features (e.g., dropped age when correlated with experience).

🔹 Transformation

Standardize numerical features (StandardScaler).

One-hot encode categorical features (OneHotEncoder).

Return processed feature set + target (salary).

📊 Workflow
flowchart TD
    A[Raw CSV Data] --> B[Clean Data]
    B --> C[Feature Engineering]
    C --> D[Remove Multicollinearity]
    D --> E[Scale Numeric Features]
    D --> F[Encode Categorical Features]
    E --> G[Final ML-ready Dataset]
    F --> G

🚀 Usage

Clone the repo and install dependencies:

git clone <your-repo-url>
cd repo-folder
pip install -r requirements.txt


Run the preprocessing script:

python preprocess.py


Use the final processed dataset:

X → feature matrix (scaled + encoded).

y → target labels (salary).

Example:

from preprocess import transform_data, data

X, y = transform_data(data)
print(X.head())
print(y.head())

📦 Dependencies

pandas

numpy

requests

scikit-learn

Install with:

pip install pandas numpy requests scikit-learn
