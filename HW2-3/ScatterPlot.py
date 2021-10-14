import pandas as pd
import matplotlib.pyplot as plt

# Load dataframe using pandas
df = pd.read_csv("HW2-3/HW2-3.txt", header=None)

# Extract u and v coordinates from dataframe
u = df.iloc[0]
v = df.iloc[1]

# Show the scatter plot
plt.scatter(u, v)
plt.show()
