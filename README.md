# Tax Analyst Sim: Synthetic Brokerage Tax Data Generator

Generates a fully fictional brokerage tax dataset and a corrupted copy with a
ground-truth answer key, so you can practice tax operations analysis without
using real customer data.

**Phase 1: generate_dataset.py**
Creates five linked CSVs: accounts, transactions, cost_basis, tax_forms and
withholding. It simulates ~750 accounts and ~14,000 transactions for one tax
year, including FIFO tax lots, wash-sale adjustments, dividends and interest,
W-9/W-8 documentation, backup and NRA withholding, and year-end 1099-B, 1099-DIV,
1099-INT, 1099-R and 1042-S forms. It runs its own QA report before export.

**Phase 2: inject_errors.py**
Copies the clean data and plants about 100 realistic errors: missing or wrong
cost basis, mismatched 1099 totals, duplicate transactions, expired W-8s,
missing withholding, and so on. Every error is recorded in `error_log.csv`, so
you can score your own detection queries on what they caught and missed.

Both scripts are seeded and reproducible. Change the seed to generate a
different dataset with the same structure.

All data is fictional. This is for learning, not tax advice.
