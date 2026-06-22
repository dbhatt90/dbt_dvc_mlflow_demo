import pandas as pd

# Note: this file has two sheets (Year 2009-2010 and Year 2010-2011)
xls = pd.ExcelFile("online_retail_II.xlsx")
print(xls.sheet_names)  # check sheet names

df1 = pd.read_excel(xls, sheet_name=0)
df2 = pd.read_excel(xls, sheet_name=1)

df = pd.concat([df1, df2], ignore_index=True)
df.to_csv("raw_orders.csv", index=False)