from models.MyModel import MyModel
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


class MyLogisticRegression(MyModel):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def init_model(self, C=10, max_iterations=10000):
        self.model = LogisticRegression(C=C, max_iter=max_iterations)

    def cross_validate_penalty(self, C_range=[1, 10, 100, 1000], k_fold_nb=5, max_iterations=10000):
        print("> C Cross validation (range:", C_range, ")")
        mean_scores = []; std_scores = []

        for C in C_range:
            print("\t> For C = %.2f..." % C)
            tmp_model = LogisticRegression(C=C, max_iter=max_iterations)
            scores = cross_val_score(tmp_model, self.X_train, self.y_train, cv=k_fold_nb, scoring='f1')
            mean_scores.append(np.array(scores).mean())
            std_scores.append(np.array(scores).std())

        print("> C Cross validation done")
        plt.figure()
        plt.errorbar(C_range, mean_scores, yerr=std_scores)
        plt.title("C Penalty Cross Validation")
        plt.xlabel("C value")
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
        print("> Classification report:")
        print(classification_report(self.y_test, y_pred, target_names=["y = False", "y = True"]))

        print("> Intercept:")
        print(self.model.intercept_)
        print("> Coefficients:")
        print(self.model.coef_)

    def plot_roc_curve(self, display_plot=True):
        """Plot ROC curve and ROC area of the model"""
        decision_fct = self.model.decision_function(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test, decision_fct)
        plt.figure(123)
        plt.plot(fpr, tpr, label='Logistic regression (area = %0.2f)' % auc(fpr, tpr))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression')
        plt.legend(loc="lower right")
        if display_plot:
            plt.show()