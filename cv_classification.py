from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import numpy as np

def cross_val_classification_report(model,X,y,cv=5, random_state=None):
    """Computes the classification report over the all dataset using cross validation.
    params: 
    - cv may be an integer or a splitter. If it's a splitter, random_state is ignored.
    """
    if X is None or y is None or model is None:
        print("Params error! model?, X?, y?")
        return
    
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
    n_classes = len(list(set(y)))
    classes = sorted(list(set(y)))
    cl_scores = np.zeros((n_classes, 5))

    for train_index, test_index in cv.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train,y_train)
        y_p = model.predict(x_test)

        #Accuracy
        cm = confusion_matrix(y_test,y_p, labels=classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        a = cm.diagonal()

        #Precision recall f1 support
        p, r, f1, s = precision_recall_fscore_support(y_test,y_p, labels=classes)    

        for i in range(n_classes):
            cl_scores[i][0] += a[i]*s[i]
            cl_scores[i][1] += p[i]*s[i]
            cl_scores[i][2] += r[i]*s[i]
            cl_scores[i][3] += f1[i]*s[i]
            cl_scores[i][4] += s[i]

    print('Class\t\tAccuracy\tPrecision\tRecall\t\tF1\t\tSupport')

    # For each class
    for i in range(n_classes):
        cl_scores[i][:4] /= cl_scores[i][4]
        s = str(classes[i]) + '\t\t'
        for j in range(4):
            s += f'{cl_scores[i][j]:.2f}\t\t'
        # Support
        s += f'{cl_scores[i][4]:.0f}'
        print(s)

    # Overall
    oa_scores = np.zeros(5)
    oa_scores[:4] = np.sum(cl_scores[:, :4] * cl_scores[:, 4].reshape(-1,1), axis=0) / np.sum(cl_scores[:, 4])
    oa_scores[4] = np.sum(cl_scores[:, 4])

    print()
    s = 'Weighted:\t'
    for i in range(4):
        s += f'{oa_scores[i]:.2f}\t\t'
    # Support
    s += f'{oa_scores[4]:.0f}'
    print(s)