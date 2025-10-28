import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump
import tqdm

DATA_RAW = Path("../translator/data/raw")
MODEL_DIR = Path("../translator/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    X, y = [], []
    for label_path in DATA_RAW.iterdir():
        if not label_path.is_dir():
            continue
        label = label_path.name
        for npy_file in label_path.glob("*.npy"):
            lm = np.load(npy_file)
            X.append(lm)
            y.append(label)
    return np.array(X), np.array(y)

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"{X.shape[0]} samples, {len(np.unique(y))} classes")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training KNN (k=3)...")
    clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
    clf.fit(X_scaled, y)

    model_path = MODEL_DIR / "asl_knn.pkl"
    dump((clf, scaler), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()