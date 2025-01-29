from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from typing import Optional, Tuple
from collections import Counter

# We divide track into short samples, and train boostring on them. To predict we take 
# an majority category over samples of single track. 
# It serves several purposes: 
# 1. Regularization (i.e. decreases varience)
# 2. TODO: Write reasoning

# ChatGPT query:
# Write a function with following signature

# def predict_catboost(model, idx: np.array, X: np.array, y: Optional[np.array] ):
#     pass
 
# It receives catboost model the np.array with features and np.array with indices. Samples with same index correspond to the same audio track. Run catboost on all samples and aggregate results for the same index using majority voting. Also aggregate labels (they will be the same for same indices. Return arrays idx, pred, and y if it was given, sorted by idx

def predict_catboost(model, idx: np.array, X: np.array, y: Optional[np.array] = None) \
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Predicts labels for each row in X using the provided CatBoost model, then aggregates
    predictions and (optionally) labels by majority vote (for the predictions) and single
    unique value (for the labels) for each unique index in idx.
    
    Parameters
    ----------
    model : catboost.CatBoost
        Trained CatBoost model with a .predict method.
    idx : np.ndarray
        Array of indices, where samples with the same index correspond to the same audio track.
    X : np.ndarray
        Feature matrix corresponding to the samples.
    y : Optional[np.ndarray], default=None
        Array of labels. If provided, it is assumed that all samples with the same idx have the
        same label.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        - Unique indices sorted in ascending order.
        - Predicted labels (majority vote) for each unique index.
        - Labels corresponding to each unique index (if y was given), otherwise None.
    """
    # 1. Predict on all samples
    predictions = model.predict(X)[:, 0] # Catboost returns array of shape (n_samples, 1) for some reason.

    # 2. Identify unique indices in sorted order
    unique_idx = np.unique(idx)

    # 3. Aggregate predictions (and labels if present) by idx
    aggregated_preds = []
    aggregated_labels = [] if y is not None else None

    for i in unique_idx:
        rows = np.where(idx == i)[0]
        pred_slice = predictions[rows]
        most_common_label = Counter(pred_slice).most_common(1)[0][0]
        aggregated_preds.append(most_common_label)

        # For y (labels), assume they're all the same for these rows
        if y is not None:
            aggregated_labels.append(y[rows][0])  # or np.unique(y[rows])[0]

    # Convert lists to numpy arrays
    aggregated_idx = unique_idx
    aggregated_preds = np.array(aggregated_preds)
    if aggregated_labels is not None:
        aggregated_labels = np.array(aggregated_labels)

    return aggregated_idx, aggregated_preds, aggregated_labels


def train_and_evaluate_catboost(
    idx_eval: np.ndarray, 
    X_train: np.ndarray, 
    X_eval: np.ndarray, 
    y_train: np.ndarray, 
    y_eval: np.ndarray, 
    bagging_temperature=1, 
    feature_weights=None,
    cat_features = None
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
        verbose=50,              # Real-time output every 50 iterations
        random_seed=42,           # For reproducibility
        bagging_temperature=bagging_temperature,
        feature_weights=feature_weights,
        cat_features=cat_features,
        one_hot_max_size=20
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=(X_eval, y_eval),
        use_best_model=True,  # Use the best iteration during training
        plot=True             # Show training progress plot (if supported)
    )
    
    # Evaluate the model
    _, y_pred, y_true = predict_catboost(model, idx_eval, X_eval, y_eval)
    eval_accuracy = accuracy_score(y_true, y_pred)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
    
    return model, eval_accuracy


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