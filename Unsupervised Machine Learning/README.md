## Unsupervised Machine Learning Overview

Unsupervised learning is a class of machine learning techniques used to discover hidden patterns or intrinsic structures in data without using labeled outputs. These algorithms operate solely on input data and aim to group, compress, or otherwise understand the data's underlying organization.

This section highlights several fundamental unsupervised learning techniques, each serving a different purpose:

### K-Means Clustering
K-Means is a centroid-based clustering algorithm that partitions data into *k* clusters. It initializes *k* cluster centers, assigns each data point to the nearest center, and iteratively updates the centers to minimize intra-cluster variance. It's efficient and widely used, though sensitive to initialization and assumes spherical clusters.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN groups together points that are closely packed and marks points that lie alone in low-density regions as outliers. Unlike K-Means, it does not require specifying the number of clusters in advance and can identify arbitrarily shaped clusters. It is particularly useful for spatial data and handling noise.

### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible. It does so by finding orthogonal axes (principal components) that capture the maximum variance in the data. PCA is commonly used for visualization, noise reduction, and preprocessing.

### Image Compression with the Singular Value Decomposition (SVD)
SVD is a matrix factorization technique that decomposes a matrix into three components: \( U \), \( \Sigma \), and \( V^T \). In the context of image compression, SVD allows us to approximate the original image by keeping only the most significant singular values, thus reducing storage without sacrificing much visual quality. It’s a powerful tool for lossy compression and data approximation.