import matplotlib.pyplot as plt
import mplcursors
# create a sample dataframe
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\dl_proj\dl_project\notebooks\merged_df.csv')
# create the scatter plot
fig, ax = plt.subplots()
sc = ax.scatter(df['umap_0'], df['umap_1'], c=df['OD'], cmap='viridis')
plt.colorbar(sc)

# add interactivity
def on_select(sel):
    index = sel.target.index
    path = df.loc[index, 'full_path']
    # open the image at the selected path, e.g. using Pillow
    img = np.load(path)
    print(img[0,...].shape)
    # plt.imshow(img[0,...])
    # plt.show()

mplcursors.cursor(sc).connect("add", on_select)

plt.show()
