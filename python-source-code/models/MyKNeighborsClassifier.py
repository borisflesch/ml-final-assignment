from models.MyModel import MyModel
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


class MyKNeighborsClassifier(MyModel):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def init_model(self, n=5):
        self.model = KNeighborsClassifier(n_neighbors=n, weights='uniform')

    def cross_validate_penalty(self):
        print("No cross-validation available for kNN")

    def cross_validate_n(self, n_range=[1, 10, 100, 1000], k_fold_nb=5):
        print("> n cross validation (range:", n_range, ")")
        mean_scores = []; std_scores = []

        for n in n_range:
            print("\t> For n = %d..." % n)
            tmp_model = KNeighborsClassifier(n_neighbors=n, weights='uniform')
            scores = cross_val_score(tmp_model, self.X_train, self.y_train, cv=k_fold_nb, scoring='f1')
            mean_scores.append(np.array(scores).mean())
            std_scores.append(np.array(scores).std())

        print("> n cross validation done")
        plt.figure()
        plt.errorbar(n_range, mean_scores, yerr=std_scores)
        plt.title("n (# neighbors) Cross Validation")
        plt.xlabel("n (# neighbors)")
        plt.ylabel("F1-Score")
        plt.show()

    def confusion_matrix(self):
        print("> Confusion matrix:")
        plt.figure()
        disp = plot_confusion_matrix(self.model, self.X_test, self.y_test, display_labels=["y = False", "y = True"], cmap=plt.cm.Blues,
                                     values_format='d')
        disp.ax_.set_title("Confusion matrix")
        plt.show()

    def print_report(self):
        y_pred = self.model.predict(self.X_test)
        print("> Scores:")
        print(classification_report(self.y_test, y_pred, target_names=["y = False", "y = True"]))

    def plot_roc_curve(self, display_plot=True):
        """Plot ROC curve and ROC area of the model"""
        y_pred_proba = self.model.predict_proba(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
        plt.figure(123)
        plt.plot(fpr, tpr, label='kNN (area = %0.2f)' % auc(fpr, tpr))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - kNN')
        plt.legend(loc="lower right")
        if display_plot:
            plt.show()