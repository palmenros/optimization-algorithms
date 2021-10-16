import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load dataframe using pandas
df = pd.read_csv("mnist_train.csv", header=None)

for i in range(0, 10):
    # Filter the dataframe for the specific digit we are searching
    dataframe_digit_i = df[df.iloc[:, 0] == i]

    # Pick a random representative from the dataframe
    random_representative = dataframe_digit_i.sample()
    representative_index = random_representative.index[0]

    image_data = random_representative.iloc[0, 1:].to_numpy().reshape((28, 28))

    print(f'Plotting as representative of digit #{i} image with index {representative_index}')

    plt.imshow(image_data, interpolation='nearest')
    plt.show()