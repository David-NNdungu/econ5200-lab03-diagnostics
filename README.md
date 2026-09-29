# Pipeline Health Check — EDA, Corruption & Distribution Shift

## Objective

Diagnose data corruption in a country-year panel and check whether incoming data have shifted from the training sample.

## Methodology

- Inspected structure, distributions, duplicate country-year keys, and value ranges.
- Corrected sign and unit errors, removed unrecoverable trade values, and deduplicated country-years.
- Compared training and inference distributions with plots and Population Stability Index (PSI).
- Compared manual checks with automated profiling and wrote reusable validation functions in `eda_utils.py`.
- Built an interactive notebook dashboard for distributions, PSI thresholds, constraints, and before/after summaries.

## Key findings

- Identified five planted data quality issues: negative GDP, life expectancy in months, duplicate keys, inconsistent trade percentages, and mixed GDP units.
- The cleaned panel contains 230 country-year rows; 30 rows were removed through duplicate handling or unrecoverable trade values.
