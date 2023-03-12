import matplotlib.pyplot as plt
import mplcursors
# create a sample dataframe
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\dl_proj\dl_project\notebooks\merged_df.csv')
# create the scatter plot and image subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sc = ax1.scatter(df['umap_0'], df['umap_1'], c=df['OD'], cmap='viridis')
plt.colorbar(sc)

# add interactivity
def on_select(sel):
    index = sel.index
    path = df.loc[index, 'full_path']
    # open the image at the selected path in a new subplot
    img = np.load(path)
    ax2.imshow(img[0,...])
    ax2.set_axis_off()
    plt.show(block=False) # set block=False to avoid blocking the event loop

mplcursors.cursor(sc).connect("add", on_select)

plt.show()
