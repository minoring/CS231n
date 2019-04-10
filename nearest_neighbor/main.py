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

# Now that we have all images stretched out as rows,
# here is how we could train and evaluate a classifier.

# Create a Nearest Neighbor classifier class
nn = NearestNeighbor() 

# Train the classifier on the training images and labels
nn.train(Xtr_rows, Ytr)

# Predict labels on the test images
Yte_predict = nn.predict(Xte_rows)

# And now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print(f'accuracy: {np.mean(Yte_predict == Yte)}')
