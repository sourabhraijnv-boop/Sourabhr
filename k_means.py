import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        dis = []
        for j in range(centroids.shape[0]):
            l_d = np.linalg.norm(X[i] - centroids[j])
            dis.append(l_d)

        idx[i] = np.argmin(dis)

    return idx
def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        points = X[idx == i]
        centroids[i] = np.mean(points, axis=0)

    return centroids

def run_kmeans(X, intial_centroids, max_iter=10):

    m, n = X.shape
    K = intial_centroids.shape[0]
    centroids = intial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    for i in range(max_iter):
        print("K-means iteration %d/%d" % (i, max_iter-1))
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    
    return centroids, idx

def kMean_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids

original_img = plt.imread(r"C:\Users\soura\pic.jpg")


print("Shape of original img is:", original_img.shape)
# give metrices of m X 3 where m is number of pixels 1079x1920
original_img = original_img/255
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

K = 25
max_itr = 10
initial_centroids = kMean_init_centroids(X_img, K)
centroids, idx = run_kmeans(X_img, initial_centroids, max_itr)
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])
# find closest color for each pixel
idx = find_closest_centroids(X_img, centroids)
# assign centroid color to the pixel
X_recoverd = centroids[idx, :]
X_recoverd = np.reshape(X_recoverd, original_img.shape)

fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()

ax[1].imshow(X_recoverd)
ax[1].set_title('Compressed')
ax[1].set_axis_off()

plt.show()