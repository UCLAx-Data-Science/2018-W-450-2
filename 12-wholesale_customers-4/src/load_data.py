import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


customer_df = pd.read_csv('data/Wholesale_customers_data.csv')
customer_df.drop(['Channel', 'Region'], axis=1, inplace=True)

scaler = StandardScaler()
scaler.fit(customer_df)
customer_sc = scaler.transform(customer_df)
customer_sc_df = pd.DataFrame(customer_sc, columns=customer_df.columns)

scaler = StandardScaler()
customer_log_df = np.log(customer_df + 1)
scaler.fit(customer_log_df)
customer_log_sc = scaler.transform(customer_log_df)
customer_log_sc_df = pd.DataFrame(customer_log_sc, columns=customer_df.columns)
customer_final_df = customer_log_sc_df.drop([65, 66, 128, 154, 75], axis=0)
