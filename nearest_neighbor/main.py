from get_CIFAR10_images import load_cifar10_data
from NearestNeighbor import NearestNeighbor

data_dir = 'cifar-10-batches-py'
Xtr, Ytr, Xte, Yte = load_cifar10_data(data_dir)

print(Xtr.shape)
print(Ytr.shape)
print(Xte.shape)
print(Yte.shape)

## Flatten out all images to be one-dimensional

# Xtr_rows becomes 50000 * 3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

Xval_rows = Xtr_rows[:1000, :]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :]
Ytr =Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [3, 5, 10, 20, 50, 100]:
    # Use a particular value of k and evaluation on validation data
    # Now that we have all images stretched out as rows,
    # here is how we could train and evaluate a classifier.

    # Create a Nearest Neighbor classifier class
    nn = NearestNeighbor() 

    # Train the classifier on the training images and labels
    nn.train(Xtr_rows, Ytr)

    # Predict labels on the test images
    # here we assume a modified NearestNeighor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)

    print(f'accuracy: {acc}')
    
    validation_accuracies.append((k, acc))