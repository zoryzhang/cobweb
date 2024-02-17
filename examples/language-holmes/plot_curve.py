import sys
import pandas as pd
import matplotlib.pyplot as plt

"""
To see the evolution properly, 
the plots need to show the averaged data in segments instead of the data after each train-and-test.
"""
window = 100

# The data file specified in the commend line
data_file = sys.argv[1]

df = pd.read_csv(data_file)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

moving_avgp = df['prob_word'].rolling(window=window).mean()
ax1.plot(moving_avgp, color='green', label='cobweb')
ax1.set_title('{}-Instance Moving Average of P(word|context)'.format(window))
ax1.set_ylabel('P(word|context)')

moving_avga = df['correct'].rolling(window=window).mean()
ax2.plot(moving_avga, color='green', label='cobweb')
ax2.set_title('{}-Instance Moving Average of Accuracy'.format(window))
ax2.set_ylabel('Accuracy')

plt.legend()
plt.tight_layout()
plt.show()

