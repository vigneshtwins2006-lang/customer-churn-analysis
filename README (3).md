# Customer Churn Analysis (Python)

End-to-end churn analysis: data cleaning, churn rate, Customer Lifetime Value (CLV), MRR loss, segmentation by contract / tenure / payment method, charts (bar plots, boxplots, heatmaps), business recommendations and an auto-generated PDF report.

## Run
```bash
pip install pandas numpy matplotlib seaborn reportlab pillow jupyter
python churn_analysis.py raw_churn.csv          # or open Customer_Churn_Analysis.ipynb
```
Outputs go to `outputs/`: `Customer_Churn_Analysis_Report.pdf`, `cleaned_churn_data.csv`, `churn_by_segment.csv`, `key_metrics.csv`, `charts/`.

## Metric definitions
- Churn rate = churned / total customers
- Monthly churn = churn events / total customer-months (sum of tenure); lifespan = 1 / monthly churn
- CLV = ARPU x average lifespan
- MRR loss = sum of monthly charges of churned customers

`demo_output_synthetic_data/` shows the output on a synthetic dataset (`make_sample_data.py`), not real customer data.
