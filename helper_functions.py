import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    
    Parameters:
    - cm : Confusion matrix array
    - classes : List of class names
    - title : Plot title (default: 'Confusion matrix')
    - cmap : Color map for the plot (default: plt.cm.Blues)
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def train_classifier(clf, param_grid, X_train, y_train, X_test, y_test):
    """
    Trains a classifier using GridSearchCV with Stratified K-Fold cross-validation,
    evaluates its performance on test data, and plots a confusion matrix.

    Parameters:
    - clf : Classifier object
    - param_grid : Parameter grid for GridSearchCV
    - X_train : Training features
    - y_train : Training labels
    - X_test : Test features
    - y_test : Test labels
    
    Returns:
    - best_clf : Best classifier model based on GridSearchCV
    """
    skf = StratifiedKFold(n_splits=5)
    
    # Grid search for finding the best classifier
    grid_search = GridSearchCV(clf, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    # Print best parameters found by GridSearchCV
    print("Best parameters = ", grid_search.best_params_)

    # Evaluate classifier on test data
    y_pred = best_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Compute and print ROC AUC score
    print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
    
    # Compute confusion matrix and plot it
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Non-Fraud', 'Fraud'], title=f'Confusion Matrix - {best_clf.__class__.__name__}')
    
    return best_clf
