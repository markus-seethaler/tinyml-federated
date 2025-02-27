import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('federated_metrics.csv')

# Create figure and axis objects with a certain size
fig, ax1 = plt.subplots(figsize=(12, 6))

# Set the background grid
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot accuracy on primary y-axis
line1 = ax1.plot(df['Round'], df['Accuracy'], label='Accuracy (%)',
                 color='#2E86C1', linewidth=2)
ax1.set_xlabel('Federated Learning Rounds', fontsize=10)
ax1.set_ylabel('Accuracy (%)', color='#2E86C1', fontsize=10)
ax1.tick_params(axis='y', labelcolor='#2E86C1')

# Create second y-axis for loss
ax2 = ax1.twinx()
line2 = ax2.plot(df['Round'], df['TestLoss'], label='Test Loss',
                 color='#E74C3C', linewidth=2, linestyle='--')
line3 = ax2.plot(df['Round'], df['TrainingLoss'], label='Training Loss',
                 color='#27AE60', linewidth=2, linestyle=':')
ax2.set_ylabel('Loss', color='#E74C3C', fontsize=10)
ax2.tick_params(axis='y', labelcolor='#E74C3C')

# Customize x-axis ticks
x_tick_spacing = 30  # Adjust this value to change spacing
max_round = df['Round'].max()
x_ticks = np.arange(0, max_round + x_tick_spacing, x_tick_spacing)
plt.xticks(x_ticks, rotation=45)

# Add title with extra padding
plt.title('Federated Learning Metrics Over Training Rounds',
          pad=50, fontsize=12, fontweight='bold')

# Combine legends and position between title and plot
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, frameon=True, facecolor='white', edgecolor='none')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high DPI
plt.savefig('federated_learning_metrics.png', dpi=300, bbox_inches='tight')
plt.close()