import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
import helper

PREPROC_DIR = "D:/FYP/DATA_preproc"

results = []

# Regularization parameter (lambda)
alpha = 0.001   # Small value since data is large (tune if needed)

# Iterate over 18 subjects
for s_id in range(1, 19):
    # Load subject data
    eeg, env, _, _ = helper.subject_eeg_env_ab(PREPROC_DIR, s_id)
    
    n_samples = eeg.shape[0]
    train_size = int(0.8 * n_samples)
    
    X_train, X_test = eeg[:train_size], eeg[train_size:]
    y_train, y_test = env[:train_size], env[train_size:]
    
    # Fit Lasso model
    model = Lasso(alpha=alpha, max_iter=10000)  # increase max_iter for convergence
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Pearson correlation
    corr, _ = pearsonr(y_test, y_pred)
    
    print(f"Subject {s_id}: Pearson correlation = {corr:.4f}")
    results.append((s_id, corr))

# Save to CSV with model name
df = pd.DataFrame(results, columns=["s_id", "pearson_corr"])
csv_filename = "lasso_results.csv"
df.to_csv(csv_filename, index=False)

print(f"\nSaved results to {csv_filename}")
