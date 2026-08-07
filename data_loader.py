import ssl
import os
import pickle
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Bypass SSL for automated dataset downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Directory to store the cached data files
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_dataset(name="mnist"):
    img_size = 28 if name == "mnist" else 32
    
    # Define a unique cache filename for this dataset and size
    cache_path = os.path.join(CACHE_DIR, f"{name}_{img_size}.pkl")
    
    # Check if a cached version of the processed data already exists
    if os.path.exists(cache_path):
        print(f"  [CACHE] Loading {name.upper()} from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # If no cache is found, process the data as usual
    print(f"  [DOWNLOAD] Processing {name.upper()} for the first time...")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    if name == "mnist":
        tr = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        te = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "svhn":
        tr = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        te = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        num_classes = 10
    elif name == "cifar100":
        tr = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        te = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    
    X_train = np.array([i.numpy() for i, _ in tr])
    y_train = np.array([l for _, l in tr])
    X_test = np.array([i.numpy() for i, _ in te])
    y_test = np.array([l for _, l in te])
    
    data_bundle = (X_train, y_train, X_test, y_test, num_classes)

    # Save the processed data to the cache directory for future use
    with open(cache_path, "wb") as f:
        pickle.dump(data_bundle, f)
    print(f"  [SUCCESS] {name.upper()} cached to {cache_path}")
    
    return data_bundle

def iid_partition(X, y, N, samples_per_user, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:N * samples_per_user]
    X_pool, y_pool = X[idx], y[idx]
    return [(X_pool[i*samples_per_user:(i+1)*samples_per_user], 
             y_pool[i*samples_per_user:(i+1)*samples_per_user]) for i in range(N)]