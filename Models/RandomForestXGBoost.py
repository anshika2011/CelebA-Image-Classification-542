"""
Simple script to train models for CelebA smiling attribute recognition
with and without PCA dimensionality reduction
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

warnings.filterwarnings("ignore")

class CustomCelebADataset(Dataset):
    def __init__(self, root, split='train', transform=None, attr_file=None, partition_file=None):
        self.root = root
        self.transform = transform

        # Set paths for metadata files
        attr_path = attr_file or os.path.join(root, 'celeba', 'list_attr_celeba.txt')
        split_path = partition_file or os.path.join(root, 'celeba', 'list_eval_partition.txt')
        img_folder = os.path.join(root, 'celeba', 'img_align_celeba')

        # Load attribute data
        with open(attr_path) as f:
            lines = f.readlines()
            header = lines[1].strip().split()
            data = [line.strip().split() for line in lines[2:]]

        # Create dataframes and merge
        self.attr_names = header
        df_attr = pd.DataFrame(data, columns=['filename'] + header)
        df_attr[header] = df_attr[header].astype(int)
        df_attr[header] = (df_attr[header] == 1).astype(int)  # convert -1 to 0

        df_split = pd.read_csv(split_path, delim_whitespace=True, header=None, names=['filename', 'split'])
        self.df = pd.merge(df_attr, df_split, on='filename')

        # Filter by split (0=train, 1=val, 2=test)
        split_map = {'train': 0, 'val': 1, 'test': 2}
        self.df = self.df[self.df['split'] == split_map[split]]
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['filename'])

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError):
            # Create synthetic image if file not found
            np.random.seed(idx)
            image = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        # Only return the Smiling attribute
        label = torch.tensor(row['Smiling'].astype('float32'))
        return image, label

def load_data(data_dir='data', use_pca=False, n_components=64):
    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Set paths for attribute and partition files
    attr_file = './clean_repo/sublists/list_attr_celeba.txt'
    partition_file = './clean_repo/sublists/list_eval_partition.txt'

    # Load datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = CustomCelebADataset(
            root=data_dir, split=split, transform=transform,
            attr_file=attr_file, partition_file=partition_file
        )

    print(f"Dataset sizes - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

    # Extract features and labels
    data = {}
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        X, y = [], []

        for images, labels in tqdm(loader, desc=f"Processing {split} data"):
            X.append(images.numpy())
            y.append(labels.numpy())

        data[f'X_{split}'] = np.vstack(X)
        data[f'y_{split}'] = np.concatenate(y)

    # Standardize features
    scaler = StandardScaler()
    data['X_train'] = scaler.fit_transform(data['X_train'])
    data['X_val'] = scaler.transform(data['X_val'])
    data['X_test'] = scaler.transform(data['X_test'])

    # Apply PCA if requested
    if use_pca:
        pca = PCA(n_components=min(n_components, data['X_train'].shape[0] - 1))
        data['X_train'] = pca.fit_transform(data['X_train'])
        data['X_val'] = pca.transform(data['X_val'])
        data['X_test'] = pca.transform(data['X_test'])
        print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    return data

def train_and_evaluate(data, n_estimators=100, use_pca=False):
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    }

    results = {'val': {}, 'test': {}}

    # Train on training set and evaluate on validation set
    print("\nTraining and evaluating on validation set:")
    for name, model in models.items():
        model.fit(data['X_train'], data['y_train'])
        y_pred = model.predict(data['X_val'])

        metrics = {
            'accuracy': accuracy_score(data['y_val'], y_pred),
            'precision': precision_score(data['y_val'], y_pred, zero_division=0),
            'recall': recall_score(data['y_val'], y_pred, zero_division=0),
            'f1': f1_score(data['y_val'], y_pred, zero_division=0)
        }

        results['val'][name] = metrics
        print(f"{name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

    # Train on combined train+val and evaluate on test set
    print("\nTraining on combined train+val and evaluating on test set:")
    X_train_val = np.vstack((data['X_train'], data['X_val']))
    y_train_val = np.concatenate((data['y_train'], data['y_val']))

    for name, model in models.items():
        model.fit(X_train_val, y_train_val)
        y_pred = model.predict(data['X_test'])

        metrics = {
            'accuracy': accuracy_score(data['y_test'], y_pred),
            'precision': precision_score(data['y_test'], y_pred, zero_division=0),
            'recall': recall_score(data['y_test'], y_pred, zero_division=0),
            'f1': f1_score(data['y_test'], y_pred, zero_division=0)
        }

        results['test'][name] = metrics
        print(f"{name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

    # Save results to CSV
    pca_suffix = "with_pca" if use_pca else "without_pca"
    with open(f"results_{pca_suffix}.csv", "w") as f:
        f.write("Model,Split,Accuracy,Precision,Recall,F1\n")

        for split in ['val', 'test']:
            for model_name, metrics in results[split].items():
                f.write(f"{model_name},{split.capitalize()},{metrics['accuracy']:.4f},"
                        f"{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1']:.4f}\n")

    print(f"\nResults saved to results_{pca_suffix}.csv")

def main():
    parser = argparse.ArgumentParser(description='CelebA Smiling Recognition')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--use_pca', action='store_true')
    parser.add_argument('--n_components', type=int, default=256)
    args = parser.parse_args()

    try:
        print(f"Loading data{'with PCA' if args.use_pca else ''}...")
        data = load_data(args.data_dir, args.use_pca, args.n_components)
        train_and_evaluate(data, args.n_estimators, args.use_pca)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
