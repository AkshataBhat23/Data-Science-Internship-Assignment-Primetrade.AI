# Data Science Internship Assignment

This repository contains the analysis for the Primetrade.ai Data Science Internship assignment on **Trader Performance vs Market Sentiment**.

## Setup & How to Run

1. **Prerequisites**
   - Python 3.9+ 
   - Jupyter Notebook (`pip install notebook`)
   - Essential DS libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`
   - Bonus libraries: `scikit-learn`, `streamlit`
   
   To install requirements:
   ```sh
   pip install pandas numpy matplotlib seaborn jupyter scikit-learn streamlit
   ```

2. **Data Files**
   Ensure `fear_greed_index.csv` and `historical_data.csv` are in the same directory as the notebook `.ipynb` file.

3. **Running the Analysis & Machine Learning**
   Open the Jupyter Notebook:
   ```sh
   jupyter notebook DS_Internship_Notebook.ipynb
   ```
   Execute all cells linearly from top to bottom. It will run through data ingestion, visualizations, K-Means clustering, and train the Random Forest classification model. Note that visual boxplots utilize a Symmetrical Logarithmic (symlog) scale to accurately map extreme historical outliers.

4. **Static Charts Generation**
   You can natively export all visual findings to image `.png` formats without opening the notebook using the included python script:
   ```sh
   python generate_charts.py
   ```
   This will output 5 analytical charts right into the `output_charts` directory.

5. **Running the Interactive Dashboard (Bonus)**
   A Streamlit dashboard is included to dynamically filter and view the trader archetypes and sentiments.
   Run it via terminal:
   ```sh
   streamlit run app.py
   ```

---

## Methodology & Analysis Summary

### Data Preparation
- **Timestamps**: Parsed daily sentiment `.csv` timestamps using `%d-%m-%Y`. Transformed raw `Timestamp` in the historical data (in milliseconds) into a daily datetime format to align the datasets on `date`.
- **Metrics Evaluated**: Daily Closed PnL per Account, Trade Frequency per Account, Average Order Size in USD (used as proxy for leverage/risk size), and Long/Short Ratio per day.

### Identified Insights

1. **Performance Discrepencies**:
   - Accounts typically see slightly lower aggregate daily PnLs on "Fear" days vs "Greed" days. Win rate distribution slightly squeezes downward when markets undergo fear panics.
   
2. **Behavioral Adjustments**:
   - Trading frequency and sizing generally spike. Some retail traders dramatically increase position sizing (either catching knives or revenge trading) during Fear events, leading to more inconsistent outliers compared to Greed events.

3. **Trader Segments**:
   - **High Volume vs Low Volume**: Segmented using `Size USD` as a proxy. High volume accounts are slightly more likely to show positive consistency.
   - **Frequent vs Infrequent Traders**: More frequent traders encounter harsher drawdowns in "Fear" days than infrequent traders.

### Strategy Recommendations (Actionable Output)

1. **Context-Aware Position Sizing**
   - **Rule of Thumb**: During extended "Fear" periods, traders should proactively reduce their normal base position sizes (or leverage). Volatility tends to skew PnL negatively, resulting in larger variance. Conservative sizing defends capital against large gap-downs.

2. **Momentum vs Reversion Alignment**
   - **Rule of Thumb**: For "Frequent" traders, algorithmic tracking logic should tighten stop-losses and perhaps aggressively filter signals to reduce trade frequency on "Fear" days. Overtrading in fear conditions consistently breaks long-term capital compounding.
