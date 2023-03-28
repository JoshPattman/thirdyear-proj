from model import make_clone_model, get_xy, make_model
import numpy as np

(train_X, train_Y), (test_X, test_Y) = get_xy(6000)

model = make_clone_model()

model.fit(train_X, train_Y, epochs=20, verbose=True)

preds = model.predict(test_X, verbose=False)
num_correct = 0
for i in range(len(test_Y)):
    if np.argmax(preds[i]) == test_Y[i]:
        num_correct += 1
accuracy = 100*num_correct/test_Y.shape[0]

print("ACCURACY, %s"%accuracy)