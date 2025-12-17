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

# --- NEW STEP 4.5: CALCULATE RECONSTRUCTION ERROR (Validation) ---
# Project back to original space to see how much info was lost
X_reconstructed = pca.inverse_transform(principalComponents)
# Frobenius norm of the difference
reconstruction_loss = np.linalg.norm(X_scaled - X_reconstructed) / np.linalg.norm(X_scaled)

# --- 5. GRAPH 1: THE BIPLOT (CORRECTED) ---
plt.figure(figsize=(10, 8))

# Logic: 
# PC2 > 0 (Positive/Blue) = Historical = High Human Casualties
# PC2 < 0 (Negative/Red)  = Modern     = High Economic Cost
colors = ['#d62728' if y < 0 else '#1f77b4' for y in pca_df['PC2']]

for i, color in enumerate(['#d62728', '#1f77b4']):
    # FIX: Swapped labels to match the analysis (Red=Economic, Blue=Human)
    label = ['High Economic Cost', 'High Human Cost'][i] 
    subset = pca_df[(pca_df['PC2'] < 0) if i == 0 else (pca_df['PC2'] >= 0)]
    plt.scatter(subset['PC1'], subset['PC2'], alpha=0.6, c=color, edgecolors='k', label=label)

# Draw arrows (Existing code is fine)
scale_factor = 3.5
for i, feature in enumerate(features):
    plt.arrow(0, 0, pca.components_[0, i]*scale_factor, pca.components_[1, i]*scale_factor, 
              color='black', width=0.05, head_width=0.2) 
    plt.text(pca.components_[0, i]*scale_factor*1.15, pca.components_[1, i]*scale_factor*1.15, 
             feature, color='black', weight='bold', fontsize=12, ha='center', va='center')

plt.xlabel(f'PC1 - Overall Severity ({pca.explained_variance_ratio_[0]:.1%} Variance)')
plt.ylabel(f'PC2 - Casualty vs. Economic ({pca.explained_variance_ratio_[1]:.1%} Variance)')
plt.title('PCA of Vietnam Natural Disasters (Clustered by Impact Type)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()
plt.savefig('final_pca_biplot.png')
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
print(f"Reconstruction Loss: {reconstruction_loss:.4f} (Lower is better)")

# --- 7. GRAPH 3: TIME-SERIES PCA (CORRECTED COLORS) ---
plt.figure(figsize=(12, 6))

years = df['Start Year']
pc2_values = pca_df['PC2']
colors_time = ['#d62728' if y < 0 else '#1f77b4' for y in pc2_values] # Red=Neg, Blue=Pos

plt.scatter(years, pc2_values, alpha=0.7, c=colors_time, edgecolors='k', s=50)
plt.axhline(0, color='black', linestyle='-', linewidth=1.5)

y_min, y_max = pc2_values.min(), pc2_values.max()

# FIX: Match text color to the point color
# Historical = Positive PC2 = BLUE Points
plt.text(years.min(), y_max - 0.5, "Historical Domain\n(High Casualties)", 
         fontsize=11, color='#1f77b4', fontweight='bold', va='top') # Changed to Blue

# Modern = Negative PC2 = RED Points
plt.text(years.max() - 15, y_min + 0.5, "Modern Domain\n(High Economic Cost)", 
         fontsize=11, color='#d62728', fontweight='bold', va='bottom', ha='right')

plt.title('Evolution of Disaster Impact Nature (1953-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('PC2 Value (Nature of Impact)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

z = np.polyfit(years, pc2_values, 1)
p = np.poly1d(z)
plt.plot(years, p(years), "g--", linewidth=2, label=f'Trend (Slope: {z[0]:.4f})')
plt.legend()

plt.savefig('final_pca_time_series.png')
plt.show()