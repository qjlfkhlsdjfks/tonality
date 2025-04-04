# To create tone curve plots
import matplotlib.pyplot as plt

# Import sentiments
from main import filtered_sentiments


plt.figure(figsize=(10, 6), dpi=100)
plt.plot(filtered_sentiments)
plt.xlabel('Number of sentecnce')
plt.ylabel('Sentiment')
plt.grid()

plt.show()