from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import numpy as np
from typing import Optional, Tuple
from collections import Counter
from sklearn.model_selection import StratifiedKFold

def train_and_evaluate_catboost(
    X: np.ndarray,
    y: np.ndarray,
    verbose=True,
    train_final=True,
    **kwargs
):
    """
    Train and evaluate a CatBoost multiclass classifier using 5-fold cross-validation.
    
    Parameters:
    - X: numpy array of features
    - y: numpy array of labels
    
    Returns:
    - final_model: Trained CatBoostClassifier model on the entire dataset
    - mean_accuracy: Mean accuracy across the 5 folds
    """

    kwargs.setdefault("learning_rate", 0.1)
    kwargs.setdefault("depth", 3)
    
    # Set up stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    # Perform cross-validation
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # Initialize CatBoostClassifier
        model = CatBoostClassifier(
            iterations=500,                           
            loss_function='MultiClass',  
            eval_metric='Accuracy',  
            random_seed=42,
            verbose=100 if verbose else False,  # Set verbose=True or a numeric value if you want progress logs
            **kwargs
        )
        
        # Train on the fold
        model.fit(X_train, y_train)
        
        # Predict on the validation fold
        y_pred_fold = model.predict(X_valid)

        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_valid, y_pred_fold)
        fold_accuracies.append(fold_accuracy)

        # Collect predictions and true labels for a consolidated report
        all_y_pred.extend(y_pred_fold)
        all_y_true.extend(y_valid)

    # Compute overall metrics
    mean_accuracy = np.mean(fold_accuracies)
    if verbose:
        print(f"Mean Accuracy across 5 folds: {mean_accuracy:.4f}")
        print("\nClassification Report (aggregated across folds):")
        print(classification_report(all_y_true, all_y_pred))

    if not train_final:
        return None, mean_accuracy
    # (Optional) Train a final model on the entire dataset
    # so you have a ready-to-use classifier after cross-validation
    final_model = CatBoostClassifier(
        iterations=500,                           
        loss_function='MultiClass',  
        eval_metric='Accuracy',  
        verbose=100 if verbose else False,
        random_seed=42,
        **kwargs
    )
    final_model.fit(X, y)

    return final_model, mean_accuracy



def select_features_catboost(
    X_train: np.ndarray, 
    X_eval: np.ndarray, 
    y_train: np.ndarray, 
    y_eval: np.ndarray, 
) -> None:
    """
    Train and evaluate a CatBoost multiclass classifier.
    
    Parameters:
    - X_train: numpy array of training features
    - X_eval: numpy array of evaluation features
    - y_train: numpy array of training labels
    - y_eval: numpy array of evaluation labels
    
    Returns:
    - model: Trained CatBoostClassifier model
    - eval_accuracy: Evaluation accuracy on the validation set
    """
    # Initialize CatBoostClassifier
    model = CatBoostClassifier(
        iterations=500,          # Number of boosting iterations
        learning_rate=0.1,       # Learning rate
        depth=3,                 # Depth of the tree
        loss_function='MultiClass',  # Multiclass classification
        eval_metric='Accuracy',  # Metric for evaluation
        verbose=100,              # Real-time output every 50 iterations
        random_seed=42,           # For reproducibility
    )
    
    # # Train the model
    # model.fit(
    #     X_train, y_train,
    #     eval_set=(X_eval, y_eval),
    #     use_best_model=True,  # Use the best iteration during training
    #     plot=True             # Show training progress plot (if supported)
    # )

    model.select_features(
        X_train, y_train,
        eval_set=(X_eval, y_eval),
        features_for_select=f'0-{X_train.shape[1]-1}',
        num_features_to_select=250,
        algorithm='RecursiveByShapValues',
        steps=10,
        shap_calc_type='Regular',
        train_final_model=True
    )
    return model