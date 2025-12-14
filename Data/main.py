import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD DATA ---
df = pd.read_csv('disaster_in_vietnam.csv')

# Select only the numerical impact variables
features = ['Total Deaths', 'No. Injured', 'Total Affected', "Total Damage, Adjusted ('000 US$)"]
X = df[features].fillna(0) 

# --- 2. LOG TRANSFORMATION ---
X_log = np.log1p(X)

# --- 3. STANDARDIZATION ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# --- 4. APPLY PCA ---
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

# --- 5. GRAPH 1: THE BIPLOT (The Disaster Map) ---
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, c='#1f77b4', edgecolors='k', label='Disaster Events')

# Draw arrows
scale_factor = 3.5
for i, feature in enumerate(features):
    plt.arrow(0, 0, pca.components_[0, i]*scale_factor, pca.components_[1, i]*scale_factor, 
              color='red', width=0.05, head_width=0.2)
    plt.text(pca.components_[0, i]*scale_factor*1.15, pca.components_[1, i]*scale_factor*1.15, 
             feature, color='darkred', weight='bold', fontsize=12, ha='center', va='center')

plt.xlabel(f'PC1 - Overall Severity ({pca.explained_variance_ratio_[0]:.1%} Variance)')
plt.ylabel(f'PC2 - Casualty vs. Economic ({pca.explained_variance_ratio_[1]:.1%} Variance)')
plt.title('PCA of Vietnam Natural Disasters (Log-Transformed)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()

# SAVE GRAPH 1
plt.savefig('final_pca_biplot.png') 
print("Graph 1 saved as 'final_pca_biplot.png'")
plt.show()

# --- 6. GRAPH 2: THE SCREE PLOT (The Variance Bar Chart) ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, 3), pca.explained_variance_ratio_, marker='o', linestyle='--', color='b')
plt.bar(range(1, 3), pca.explained_variance_ratio_, alpha=0.6, color='skyblue', label='Individual Variance')

plt.title('Scree Plot: Variance Explained by Each Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.xticks([1, 2], ['PC1', 'PC2'])
plt.ylim(0, 1) # Set y-axis limit from 0 to 100%
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# SAVE GRAPH 2
plt.savefig('final_scree_plot.png')
print("Graph 2 saved as 'final_scree_plot.png'")
plt.show()

# Print Results
print("\n--- ANALYSIS RESULTS ---")
print(f"PC1 Variance: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2 Variance: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total Variance Captured: {sum(pca.explained_variance_ratio_):.2%}")