"""Run both PCA and non-PCA versions and compare results"""
import os
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run and compare CelebA models with and without PCA')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--n_components', type=int, default=256)
    args = parser.parse_args()

    # Run without PCA
    print("\n" + "="*50)
    print("Running model WITHOUT PCA")
    print("="*50)

    cmd_no_pca = [
        "python", "celeba_smile_simple.py",
        "--data_dir", args.data_dir,
        "--n_estimators", str(args.n_estimators)
    ]
    subprocess.run(cmd_no_pca)

    # Run with PCA
    print("\n" + "="*50)
    print("Running model WITH PCA")
    print("="*50)

    cmd_with_pca = [
        "python", "celeba_smile_simple.py",
        "--data_dir", args.data_dir,
        "--n_estimators", str(args.n_estimators),
        "--use_pca",
        "--n_components", str(args.n_components)
    ]
    subprocess.run(cmd_with_pca)

    # Generate comparison
    generate_report()

    print("\n" + "="*50)
    print("Comparison complete! Results saved to combined_results.csv and results_comparison.png")
    print("="*50)

def generate_report():
    """Generate a combined report with all results"""
    # Load results
    df_no_pca = pd.read_csv("results_without_pca.csv")
    df_with_pca = pd.read_csv("results_with_pca.csv")

    # Add PCA column
    df_no_pca['PCA'] = "No"
    df_with_pca['PCA'] = "Yes"

    # Combine results
    df = pd.concat([df_no_pca, df_with_pca])
    df.to_csv("combined_results.csv", index=False)

    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    # Create pivot table
    summary = df.pivot_table(
        index=['Model', 'Split'],
        columns='PCA',
        values=['Accuracy', 'F1']
    )

    print(summary)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Filter test data
    test_data = df[df['Split'] == 'Test']

    # Set up bar positions
    models = test_data['Model'].unique()
    x = np.arange(len(models))
    width = 0.35

    # Plot accuracy
    plt.subplot(1, 2, 1)
    no_pca_acc = test_data[test_data['PCA'] == 'No']['Accuracy'].values
    with_pca_acc = test_data[test_data['PCA'] == 'Yes']['Accuracy'].values

    plt.bar(x - width/2, no_pca_acc, width, label='Without PCA')
    plt.bar(x + width/2, with_pca_acc, width, label='With PCA')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.xticks(x, models)
    plt.ylim(0.7, 0.9)
    plt.legend()

    # Plot F1 score
    plt.subplot(1, 2, 2)
    no_pca_f1 = test_data[test_data['PCA'] == 'No']['F1'].values
    with_pca_f1 = test_data[test_data['PCA'] == 'Yes']['F1'].values

    plt.bar(x - width/2, no_pca_f1, width, label='Without PCA')
    plt.bar(x + width/2, with_pca_f1, width, label='With PCA')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Test F1 Score')
    plt.xticks(x, models)
    plt.ylim(0.7, 0.9)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results_comparison.png')
    print("\nResults plot saved to results_comparison.png")

if __name__ == "__main__":
    main()
