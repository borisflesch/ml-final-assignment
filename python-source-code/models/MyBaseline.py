from models.MyModel import MyModel
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


class MyBaseline(MyModel):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def init_model(self):
        self.model = DummyClassifier(strategy="most_frequent")

    def cross_validate_penalty(self):
        print("No cross-validation available for Baseline")

    def confusion_matrix(self):
        print("> Confusion matrix & scores")
        disp = plot_confusion_matrix(self.model, self.X_test, self.y_test, display_labels=["y = False", "y = True"], cmap=plt.cm.Blues, values_format='d')
        disp.ax_.set_title("Confusion matrix (baseline model, most frequent)")
        plt.show()

    def print_report(self):
        y_pred = self.model.predict(self.X_test)
        precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test, y_pred, average="macro")
        print("> Baseline scores (strategy: most frequent):")
        print("Precision (avg):", precision, ", Recall (avg):", recall, ", F1-Score (avg):", fscore)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))

    def plot_roc_curve(self, display_plot=True):
        """Plot ROC curve and ROC area of the model"""
        plt.figure(123)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="Baseline")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Baseline')
        plt.legend(loc="lower right")
        if display_plot:
            plt.show()

