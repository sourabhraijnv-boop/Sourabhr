import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import plotly.offline as py
#py.init_notebook_mode()
#output_notebook()
X = np.array([[99, -1], [98, -1], [97, -2], [101, 1], [102, 1], [103, 2]])
plt.plot(X[:,0], X[:,1], 'ro')
pca_2 = PCA(n_components=2)
pca_2.fit(X)
#print(pca_2.explained_variance_ratio_)
X_trans_2 = pca_2.transform(X)
#print(X_trans_2)
pca_1 = PCA(n_components=1)
pca_1.fit(X)
#print(pca_1.explained_variance_ratio_)
X_trans_1 = pca_1.transform(X)
#print(X_trans_1)
X_reduced_1 = pca_1.inverse_transform(X_trans_1)
#print(X_reduced_1)
x =np.array([[-0.83934975, -0.21160323],
             [0.67508491, 0.25113527],
             [-0.05495253, 0.36339613],
             [-0.57524042, 0.24450324],
             [0.58468572, 0.95337657],
             [0.5663363, 0.07555096],
             [-0.50228538, -0.65749982],
             [-0.14075593, 0.02713815],
             [0.258716, -0.26890678],
             [0.02775847, -0.77709049]])
p = figure(title='10-point scatterpoint', x_axis_label= 'x-axis', y_axis_label= 'y-axis')
p.scatter(x[:,0],x[:,1],marker='o', color = '#C00000', size=5)
p.grid.visible = False
p.grid.visible = False
p.outline_line_color = None
p.toolbar.logo = None
p.toolbar_location = None
p.xaxis.axis_line_color = "#f0f0f0"
p.xaxis.axis_line_width = 5
p.yaxis.axis_line_color = "#f0f0f0"
p.yaxis.axis_line_width = 5

#show(p)

#500 sample with 1000 features 
df = pd.read_csv("C:/Users/soura/synthetic_dataset.csv")
# randomly choose 100 pairwise(x,y) tupple of features so we can scatter plot
def get_pairs(n = 100):
    from random import randint
    i = 0
    tuples = []
    while i < 100:
        x = df.columns[randint(0,999)]
        y = df.columns[randint(0,999)]
        while x==y or (x,y) in tuples or (y,x) in tuples:
            y = df.columns[randint(0,999)]
        tuples.append((x,y))
        i+=1
    return tuples

pairs = get_pairs()
'''
fig, axs = plt.subplots(10,10, figsize = (35,35))
i = 0
for rows in axs:
    for ax in rows:
        ax.scatter(df[pairs[i][0]], df[pairs[i][1]], color = "#C00000")
        ax.set_xlabel(pairs[i][0])
        ax.set_ylabel(pairs[i][1])
        i+=1
'''
corr = df.corr()
mask = (abs(corr) > 0.5) & (abs(corr) !=1)
corr.where(mask).stack().sort_values()
'''
#2d approximation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(X_pca, columns=['principle_component_1', 'principle_component_2'])
#print(df_pca.head())
plt.scatter(df_pca['principle_component_1'], df_pca['principle_component_2'], color = "#C00000")
plt.xlabel('principle_component_1')
plt.ylabel('principle_component_2')
plt.title('PCA decomposition')
plt.show()
print(sum(pca.explained_variance_ratio_))
'''
#try in 3d
pca_3 = PCA(n_components=3).fit(df)
X_t = pca_3.transform(df)
df_pca_3 = pd.DataFrame(X_t, columns=['principle_component_1','principle_component_2','principle_component_3'])
import plotly.express as px
fig = px.scatter_3d(df_pca_3, x='principle_component_1', y='principle_component_2', z='principle_component_3').update_traces(marker = dict(color = "#C00000"))
fig.write_html("3d_scatter_plot.html")
print(sum(pca_3.explained_variance_ratio_))