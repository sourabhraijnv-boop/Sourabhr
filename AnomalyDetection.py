import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

# Generate normal data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(300, 2))  # 300 points, 2 features

# Introducing anomalies: add a few outliers
anomalies = np.random.uniform(low=-10, high=10, size=(10, 2))  # 10 anomalies

# Combine normal data with anomalies
data = np.vstack([normal_data, anomalies])

# Create corresponding y array (0 for normal, 1 for anomalies)
y = np.zeros(310)  # 300 normal points + 10 anomalies
y[300:] = 1  # Mark the last 10 points as anomalies

# Shuffle data to mix normal and anomalous points
indices = np.random.permutation(data.shape[0])
X = data[indices]
y = y[indices]

# Show a portion of X and y to verify
#print(X[:5], y[:5])  # First 5 points for preview

def estimate_gaussian(X):
    m, n = X.shape
    mu = 1/m * np.sum(X, axis=0)
    var = 1/m * np.sum((X - mu)**2, axis=0)
    return mu, var

def multivariate_gaussian_pdf(X, mean, var):

    X = np.atleast_2d(X)       # ensure shape (n_samples, d)
    mean = np.asarray(mean)
    var = np.asarray(var)
    var = np.maximum(var,1e-8)
    d = mean.shape[0]
    det_cov = np.prod(var)     # determinant of diagonal covariance
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)

    # exponent term (vectorized)
    diff = X - mean
    exponent = -0.5 * np.sum((diff ** 2) / var, axis=1)

    return norm_const * np.exp(exponent)

def select_threshold(y_val,p_val):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        prediction = (p_val < epsilon)
        tp = np.sum((prediction == 1) & (y_val==1))
        fp = np.sum((prediction == 1) & (y_val==0))
        fn = np.sum((prediction == 0) & (y_val==1))
        if tp + fp != 0:
            precision = tp/(tp+fp)
        else:
            precision = 0
        recall = tp/(tp+fn)

        if precision + recall != 0:
            f1 = (2*precision*recall)/(precision + recall)
        else:
            f1 = 0
        if f1 > best_f1 :
            best_f1 = f1
            best_epsilon = epsilon
    
    return best_epsilon, best_f1

mu, var = estimate_gaussian(X)
p_val = multivariate_gaussian_pdf(X, mu, var)
epsilon, f1 = select_threshold(y, p_val)
#print(epsilon,f1) 
outliers = p_val < epsilon
'''
plt.scatter(X[:, 0], X[:, 1], marker='x', c='b')

plt.plot(X[outliers, 0], X[outliers, 1],'ro', markersize= 10, markerfacecolor= 'none', markeredgewidth= 2)
plt.show()
'''
def generate_anomaly_data(n_train=300, n_val=200, n_features=11, anomaly_fraction=0.2, random_state=42):

    rng = np.random.RandomState(random_state)

    # Normal data ~ N(0,1)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(n_train, n_features))

    # Validation: mix of normal + anomalies
    n_anomalies = int(n_val * anomaly_fraction)
    n_normals = n_val - n_anomalies

    X_val_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normals, n_features))
    # anomalies far away from normal cluster
    X_val_anomaly = rng.normal(loc=5.0, scale=1.0, size=(n_anomalies, n_features))

    # Combine and shuffle
    X_val = np.vstack([X_val_normal, X_val_anomaly])
    y_val = np.hstack([np.zeros(n_normals), np.ones(n_anomalies)])

    # Shuffle validation set
    perm = rng.permutation(n_val)
    X_val = X_val[perm]
    y_val = y_val[perm]

    return X_train, X_val, y_val

X_train, X_val, y_val = generate_anomaly_data()

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Validation labels:", y_val.shape)
print("Number of anomalies in validation:", np.sum(y_val))

# Reduce to 2D with PCA for visualization
pca = PCA(n_components=2)
X_val_2d = pca.fit_transform(X_val)
X_train_2d = pca.transform(X_train)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
            c="blue", alpha=0.5, label="Train (Normal)")
plt.scatter(X_val_2d[y_val == 0, 0], X_val_2d[y_val == 0, 1], 
            c="green", alpha=0.6, label="Validation Normal")
plt.scatter(X_val_2d[y_val == 1, 0], X_val_2d[y_val == 1, 1], 
            c="red", alpha=0.8, label="Validation Anomaly", marker="x")

plt.title("Synthetic Anomaly Detection Data (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()


mu_high, var_high = estimate_gaussian(X_train)

p_high = multivariate_gaussian_pdf(X_train, mu_high, var_high)

p_val_high = multivariate_gaussian_pdf(X_val, mu_high, var_high)

epsilon_high, f1_high = select_threshold(y_val, p_val_high)

print('Best epsilon foud using cross validation: %e'% epsilon_high)
print('Best F1 on cross validation set : %f'% f1_high)
print('# Anomly found: %d'% sum(p_high < epsilon_high))