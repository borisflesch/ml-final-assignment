from models.MyModel import MyModel
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


class MyMLPClassifier(MyModel):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def init_model(self, n=(25,), max_iterations=10000):
        self.model = MLPClassifier(hidden_layer_sizes=n, max_iter=max_iterations)

    def cross_validate_penalty(self):
        print("No cross-validation available for Baseline")

    def cross_validate_n(self, prev_hidden_layers=(), n_range=[1, 10, 100, 1000], k_fold_nb=5, max_iterations=10000):
        # Add parameters "prev_hidden_layers" (i.e. to cross-validate a new layer's size)
        print("> n cross validation (range:", n_range, ")")
        mean_scores = []; std_scores = []

        for n in n_range:
            hls = prev_hidden_layers + (n,)
            print("\t> For hidden_layer_sizes =", hls, "...")
            tmp_model = MLPClassifier(hidden_layer_sizes=hls, max_iter=max_iterations)
            scores = cross_val_score(tmp_model, self.X_train, self.y_train, cv=k_fold_nb, scoring='f1')
            mean_scores.append(np.array(scores).mean())
            std_scores.append(np.array(scores).std())

        print("> n cross validation done")
        plt.figure()
        plt.errorbar(n_range, mean_scores, yerr=std_scores)
        plt.title("n (hidden layer sizes) Cross Validation")
        plt.xlabel("n (hidden layer sizes)")
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
        plt.plot(fpr, tpr, label='MLP (area = %0.2f)' % auc(fpr, tpr))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - MLP Classifier')
        plt.legend(loc="lower right")
        if display_plot:
            plt.show()