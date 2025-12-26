
import pandas as pd
df = pd.read_csv("data/dataset_test.csv")
print(f"Columns: {df.columns.tolist()}")
print(df[['Source_Path', 'file'] if 'file' in df.columns else ['Source_Path']].head(5))
