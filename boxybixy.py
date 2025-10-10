import matplotlib.pyplot as plt
import numpy as np

# Data
general = [0.790502, 0.799399, 0.744857]
child = [0.888453, 0.873505, 0.775865]

means = [np.mean(general), np.mean(child)]
labels = ['General Preprocessing', 'Child Preprocessing']

# Plot
plt.figure(figsize=(6,4))
bars = plt.bar(labels, means, color=['#5DADE2', '#58D68D'], width=0.5)
plt.title('Mean Pearson Correlation: General vs Child EEG Preprocessing', fontsize=12)
plt.ylabel('Mean Pearson r', fontsize=11)
plt.ylim(0.7, 0.9)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}",
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
