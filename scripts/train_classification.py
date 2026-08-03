import joblib
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from pipeline import classification_engineering

def train_classification():
    X_train, y_train, scaler, train_spline = classification_engineering('../data/training_data.csv')

    # Defining non-linear parameters for Radial basis function kernel
    parameters = {'C' : [0.1, 1, 10, 100],
                  'kernel' : ['rbf'],
                  'gamma' : ['scale', 'auto', 0.01, 0.1, 1]
                  }

    # balanced weights to ensure model doesn't miss Shutdown drop
    svc = SVC(random_state=42, class_weight='balanced')

    # Targeting F1 metric since it penalizes more on incorrect predictions on minority
    svm_search = RandomizedSearchCV(
        estimator=svc,
        param_distributions=parameters,
        n_iter=10,
        cv=5,
        scoring='f1',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    svm_search.fit(X_train, y_train)

    best_svm = svm_search.best_estimator_ # Best model from the used paramters

    print(f'Best Parameters: {svm_search.best_params_}')
    print(f'Best CV F1 Score: {svm_search.best_score_:.4f}')

    # Saving the models for replication process
    joblib.dump(best_svm, '../models/svm_shutdown_classification.joblib')
    joblib.dump(scaler, '../models/classi_scaler.joblib')
    joblib.dump(train_spline, '../models/spline_classification.joblib')

    return best_svm, scaler

best_svm, scaler = train_classification()