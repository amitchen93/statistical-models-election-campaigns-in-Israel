import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle

NUM_OF_PARTIES = 29
NUM_OF_BEST_PARTIES = 8
BIG_PARTY_ACRONYMS = [name[::-1] for name in list(["טב", "ל", "מחל", "שס", "ג", "פה", "אמת", "ודעם"])]
# BIG_PARTY_NAMES = {"טב":"ימינה","מחל":"הליכוד","פה":"כחול לבן","אמת":"העבודה גשר מרץ",
#                    "שס":"שס","ג":"יהדות התורה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה","טב":"ימינה",""}
ALL_PARTY_ACRONYMS = [name[::-1] for name in list(['אמת', 'ג', 'ודעם', 'ז', 'זך', 'טב', 'י', 'יז', 'ינ', 'יף',
                   'יק', 'יר', 'כ', 'כן', 'ל', 'מחל', 'נ', 'נז', 'ני', 'נץ', 'נק',
                   'פה', 'ףז', 'ץ', 'ק', 'קי', 'קך', 'קץ', 'שס'])]


def clear_data():
    """
    clear the un necessary data from file
    :return:
    """
    # data process
    df_2020_raw = pd.read_csv("vote_per_kalpi_2020.csv",
                              encoding='iso-8859-8',
                              index_col='שם ישוב')
    # clear the data
    df_2020_raw = df_2020_raw.drop('מעטפות חיצוניות', axis=0)
    df_2020 = df_2020_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
    df_2020 = df_2020[df_2020.columns[9:]]
    df_2020_raw = df_2020_raw.drop("זץ", axis=1)
    df_2020 = df_2020.drop("זץ", axis=1)
    return df_2020_raw, df_2020

# def create_q_vector(df, thresh=0.0325):
#     df = df[PARTY_NAMES]
#     return df  # the biggest 8 of the list

def do_PCA(df, order=2):
    pca = PCA(n_components=order)  # define PCA object
    X_pca = pca.fit_transform(df)  # fit model. Compute principal components
    return X_pca, pca.components_


def scatter_plot(X, PARTY_ALL_NAMES2):
    print(X.shape)  # (30, 2)
    fig, ax = plt.subplots()
    #X = X - X.min() + 1
    #X = (np.log2(X))
    #X = (np.log1p(X))

    X = np.array(X)
    ax.scatter(X[0, :], X[1, :])

    rev_names = [name[::-1] for name in list(PARTY_ALL_NAMES2)]

    for i, txt in enumerate(rev_names):
        ax.annotate(txt, X[:, i])

    plt.figure(figsize=(10, 10))

    #plt.ylim(0, 2000)
    #plt.xlim(0, 2000)

    plt.show()

def pc_ratio_barplot(df, comp_num,title=""):
    pca = PCA(n_components=comp_num)
    pca.fit_transform(df)
    explained_varaince = pca.explained_variance_ratio_
    df_explained = pd.DataFrame({'Princaple componenets': np.arange(comp_num)+1, 'explained variance ratio':explained_varaince})
    sns.barplot(x='Princaple componenets', y='explained variance ratio',data= df_explained)
    plt.title(title)
    plt.show()

def main():
    df_raw, df = clear_data()
    df_T = df.T
    X, c = do_PCA(df_T)
    X_T = X.T
    X_DF = pd.DataFrame({'x': X_T[0, :], 'y': X_T[1, :],
                       'group': ALL_PARTY_ACRONYMS})
    # Seaborn_scatter_parties(X_DF,"unormalized_parties.png")
    df_normalized_T = df_T / np.linalg.norm(df_T, axis=1, ord=2)[:, np.newaxis]
    X_normalized, c_normalized = do_PCA(df_normalized_T)
    X_normalized_T = X_normalized.T
    X_normalized_DF = pd.DataFrame({'x': X_normalized_T[0, :], 'y': X_normalized_T[1, :],
                         'group': ALL_PARTY_ACRONYMS})
    Seaborn_scatter_parties(X_normalized_DF, "normalized_parties.png")

    df_normalized_T = df_T / np.linalg.norm(df_T, axis=1, ord=2)[:, np.newaxis]
    X_normalized, c_normalized = do_PCA(df_normalized_T,order=3)
    X_normalized_T = X_normalized.T
    X_normalized_DF = pd.DataFrame({'x': X_normalized_T[0, :], 'y': X_normalized_T[1, :],'z':X_normalized_T[2, :]
                                    ,'group': ALL_PARTY_ACRONYMS})
    Seaborn_scatter_parties(X_normalized_DF,"normalized_parties_3D.png",True)
    pc_ratio_barplot(df_T,29,title="Original PCA compenentes")
    pc_ratio_barplot(df_normalized_T,29,title="Normalized PCA components ")





def Seaborn_scatter_parties(X_DF,name,Dim3=False):
    plt.figure(figsize=(20,10))
    if Dim3:
        palette = sns.color_palette("coolwarm", as_cmap=True)
        p1 = sns.scatterplot(data=X_DF, x="x", y="y", marker="o", hue='z',size='z',sizes=(300,1000),alpha=0.87,palette=palette)
    else:
        p1 = sns.scatterplot(data=X_DF, x="x", y="y", marker="o", color="green")
    for line in range(0, X_DF.shape[0]):
        p1.text(X_DF.x[line], X_DF.y[line], X_DF.group[line], horizontalalignment='left', size='medium',
                color='black',
                weight='semibold')
    plt.savefig(name)
    plt.show()




if __name__ == '__main__':
    main()
