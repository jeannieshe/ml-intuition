# K-Means
## What did I implement?
I implemented k-means from scratch.

## Key insight
Algorithm:
- Initialize k random centroids.
- While not converged:
  - Assignment: Assign each datum to its nearest centroid
  - Update: Update each centroid location to the mean of all its assigned data

Elbow method to identify optimal k hyperparameter:
- Use the k value located at the elbow point, which minimizes inertia without incurring diminishing returns

## Visualizations
See clean_implementation.ipynb

## Aha moment
Honestly, this exercise was really good for me to understand dimensions in numpy and how to manipulate them. I learned a few core functions, like np.newaxis, masking, fancy indexing, broadcasting, and casting a list into a numpy array instead of appending directly to a numpy array.

