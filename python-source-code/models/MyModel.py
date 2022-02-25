import abc
from joblib import dump, load
from sklearn.model_selection import train_test_split


class MyModel:
    def save_model(self, path=""):
        if not path:
            path = "saved-models/" + self.__class__.__name__ + ".joblib"

        dump(self.model, path)

    def load_model(self, path=""):
        if not path:
            path = "saved-models/" + self.__class__.__name__ + ".joblib"

        self.model = load(path)

    def split(self, test_size=0.20):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)  # random_state=42

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    @abc.abstractmethod
    def init_model(self):
        pass

    @abc.abstractmethod
    def confusion_matrix(self):
        pass

    @abc.abstractmethod
    def print_report(self):
        pass

    @abc.abstractmethod
    def plot_roc_curve(self):
        pass

    @abc.abstractmethod
    def cross_validate_penalty(self):
        pass