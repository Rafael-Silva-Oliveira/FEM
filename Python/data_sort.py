import xlsxwriter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


raw_data = pd.read_excel(
    "./Python/variations.xls", sheet_name="variations")

raw_df = pd.DataFrame(raw_data)
raw_df.rename(columns={'Unnamed: 0': 'sample'}, inplace=True)
raw_df
test_sample = raw_df['sample']
test_sample

# test_no_na = raw_df[['REF', 'ALT', '#']].apply(
#     lambda x: pd.Series(x.dropna().values))
# test_no_na

raw_df['sample'].ffill(inplace=True)
# raw_df.tail(50)

data_no_na = raw_df.dropna(axis=0, how="any")
data_no_na
data_no_na.tail(50)
# raw_df = raw_df.fillna('')
# raw_df = raw_df.groupby([0]).agg(''.join)


# final_data = data_no_na.pivot(index="sample",
#                               columns=["REF", "ALT"], values=("#"))
# final_data.columns.name

final_data_2 = (data_no_na.set_index(['sample', 'REF', 'ALT'])
                ['#']
                .unstack(['REF', 'ALT'], fill_value=0)
                )
final_data_2
sample_labels = final_data_2.index
sample_labels
# https: // stackoverflow.com/questions/48958035/pandas-convert-some-rows-to-columns-in-python
final_data_2.to_excel('saved_file_1.xlsx')

# PCA analysis

# Centering and scaling the data
# Se as nossas amostras estivessem nas colunas, colocar preprocessing.scale(final_data_2.T) para fazer a transposiçao
scaled_data = preprocessing.scale(final_data_2)
# Como alternativa podemos usar o seguinte código para centrar os dados:
# StandardScaler().fit_transform(final_data_2)
pca = PCA()
pca.fit(scaled_data)  # calcular loading scores e variaçao para cada PCA
pca_data = pca.transform(scaled_data)

# Scree plot para ver quantas componentes devem estar presentes no plot final. Este plot é usado para determinar o numero de fatores/principal components para uma analise de PCA
# Calcular % de variabilidade que cada principal component tem:
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
# Criar labels para cada principal component (PC1,PC2...tendo em conta o tamanho da variavel per_var)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of explained variability')
plt.xlabel('Principal component')
plt.title('Scree plot')
plt.show()  # grande parte da variabilidade está presente nos primeiros 9 componentes (cut-off point de 2). Estes 9 PC podem fazer uma boa representaçao geral dos dados

# Colocar as novas coordenadas (9) numa data frame onde as rows sao as samples e colunas tem PC labels
pca_data
pca_df = pd.DataFrame(pca_data, index=[sample_labels], columns=labels)
pca_df

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("PCA graph")
plt.xlabel('PC1- {0}%'.format(per_var[1]))
plt.ylabel('PC2- {0}%'.format(per_var[2]))
plt.show()
