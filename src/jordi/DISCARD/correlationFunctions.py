import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def corPlot(df, str):
    """ 
    Function to plot the correlation of numeric variables
    df: a dataset
    str: a text to include in the title and file name when saving the plot
    """

    # Compute correlation against price
    correlations = df.corr()['price'].drop('price').sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 4))
    sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    plt.title(f'Correlation of Numeric Variables Against Price - {str}', fontsize=16, fontweight='bold')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'plots/correlation/Correlation-Price-{str}', bbox_inches='tight')
    #plt.show()



def corMatrix(df, str):
    """
    Showing a heatmap correlation matrix for all numeric variables pairs
    df: a dataset
    str: a text to include in the title and file name when saving the plot
    """

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f'Correlation Heatmap - {str}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'plots/correlation/CorHeatMap-Price-{str}', bbox_inches='tight')
    #plt.show()



def pairPlot(df, features, str):
    """"
    Function to make a pairplot for all numeric values
    df: a dataset
    features: the features to plot; in our code we are targeting only the top 3
    str: a text to include in the title and file name when saving the plot
    """
    plt.figure(figsize=(6, 5))
    sns.pairplot(df, vars = features + ['price'], diag_kind='kde', corner=True)
    plt.suptitle(f"Pairplot of Price and Top 3 Features - {str}", fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(f'plots/correlation/CorPairPlot-Price-{str}', bbox_inches='tight')
    #plt.show()