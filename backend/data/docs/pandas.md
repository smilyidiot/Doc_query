# Pandas

Pandas is the standard Python library for working with tabular data. Its core
structure is the DataFrame, a two-dimensional table with labelled columns. You
load data with functions like read_csv, clean it by coercing types and dropping
invalid rows, and reshape it with groupby, pivot, merge, and resample.

Grouping and aggregation are central: groupby splits rows by a key, applies an
aggregation such as sum, mean, or count, and combines the results. Time-series
work uses a datetime index together with resample to roll data up by day,
month, or hour, and pct_change to compute period-over-period growth. Pandas
pairs naturally with NumPy for vectorised numeric operations, which makes
analytics over hundreds of thousands of rows fast and concise.
