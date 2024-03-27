import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

f1_csv_path = 'IMC_Denoise_processed_image_folders/F1_pred_df.csv'
output_path = 'IMC_Denoise_processed_image_folders/F1_pred_df.png'

df = pd.read_csv(f1_csv_path, index_col=0)

# Set seaborn style
sns.set_theme("notebook")

# Create boxplot
plt.figure(figsize=(6, 6))
sns.boxplot(data=df, palette="Set3", showfliers=False)
sns.stripplot(data=df, color='black', alpha=0.5, jitter=True) 
plt.xlabel('Protein')
plt.ylabel('F1 Score')
plt.ylim(0,1)
plt.title('F1 Scores for IMC-Denoised threshold')
plt.savefig(output_path)
