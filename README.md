# AI-Based Country Future Demographic Prediction System

A complete Streamlit machine learning project for forecasting country demographics using a World Bank-style dataset.

## Features

- Country selection
- Future year selection (2026-2050)
- Multi-category output selection:
  - Total Population
  - Male Population
  - Female Population
  - Urban Population
  - Rural Population
  - Birth Rate
  - Death Rate
- ML predictions using supervised learning (Linear Regression)
- Numerical dashboard cards and interactive charts

## Dataset

The project uses this file in the workspace:

- `P_word population data_2025/93a1ed81-ef2a-4bff-b6fc-91dadc06b1a2_Data.csv`

Supported series:

- Rural population (% of total population)
- Urban population
- Population, male
- Population, female
- Birth rate, crude (per 1,000 people)
- Death rate, crude (per 1,000 people)

## Preprocessing Pipeline

1. Load wide-format dataset
2. Remove aggregate entities (such as World/region-level totals)
3. Melt into long format
4. Parse year from columns like `2023 [YR2023]`
5. Convert `..` to missing and interpolate by country + metric
6. Pivot into structured format:
   - Year | Male | Female | Urban | RuralPercent | BirthRate | DeathRate
7. Create derived columns:
   - TotalPopulation = Male + Female
   - RuralPopulation = TotalPopulation * RuralPercent / 100

## Models

Required supervised learning models with input `Year`:

- Model 1: Output `TotalPopulation`
- Model 2: Output `BirthRate`
- Model 3: Output `DeathRate`

Additional trend model:

- Urban population model for charting and derived output

Derived estimates:

- Male Population = Total * avg(Male/Total)
- Female Population = Total * avg(Female/Total)
- Rural Population = Total * avg(Rural%)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Project Files

- `main.py`: Streamlit UI, interactions, charts, and output rendering
- `demographic_ml.py`: data preprocessing, model training, predictions
- `requirements.txt`: dependencies
# AI-Based Country Future Demographic Prediction System

This project loads the World Bank-style CSV in `P_word population data_2025`, preprocesses it into a country-year panel, trains linear regression models, and forecasts future demographic values from 2026 to 2050.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

- Select a country
- Select a future year from 2026 to 2050
- Toggle one or more output categories
- Predict and display values numerically and graphically

## Data pipeline

- Load the CSV
- Remove aggregate rows such as World and Arab World
- Clean missing values represented by `..`
- Reshape from wide format to a structured country-year table
- Derive total population from male + female
- Forecast total population, birth rate, death rate, and rural percentage with linear regression
- Derive male, female, urban, and rural populations from the predictions
