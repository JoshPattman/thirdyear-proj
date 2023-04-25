import model
import numpy as np

(xt,yt), (xT, yT) = model.get_xy(num_train_samples=6000, classes=[0,1,2,3])

print(np.unique(yt, return_counts=True))