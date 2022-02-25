from models.MyBaseline import MyBaseline
from models.MyLogisticRegression import MyLogisticRegression
from models.MyPreprocessor import MyPreprocessor
from models.MyKNeighborsClassifier import MyKNeighborsClassifier
from models.MyMLPClassifier import MyMLPClassifier
import matplotlib.pyplot as plt

part_i = True
part_ii = True

if part_i:
    # (i) predict the review polarity (where a game has been “voted up" or not by the reviewer)

    # (i)(i) Preprocessing data
    print("> Preprocessing data")
    preprocessor = MyPreprocessor()
    preprocessor.read_data('dataset/reviews_17.jl.json')
    preprocessor.preprocess_data(predict="voted_up", min_df=1, max_df=0.05)
    X, y = preprocessor.get_data()

    preprocessor.print_report()

    # print("\n=== min_df cross-validation ===")
    # min_df_range = [1, 5, 10, 15, 20, 25, 50]
    # print("> min_df cross validation (range:", min_df_range, ")")
    # preprocessor.cross_validate_min_df(min_df_range=min_df_range, max_df=1.0)  # max_df=0.150

    # print("\n=== max_df cross-validation ===")
    # max_df_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    # print("> max_df cross validation (range:", max_df_range, ")")
    # preprocessor.cross_validate_max_df(min_df=1, max_df_range=max_df_range)

    # (i)(ii) Baseline model
    print("\n=== BASELINE MODEL ===")
    baseline = MyBaseline(X, y)
    baseline.split(test_size=0.20)
    baseline.init_model()
    baseline.train()
    # baseline.confusion_matrix()
    baseline.print_report()
    baseline.plot_roc_curve(display_plot=False)

    # (i)(iii) Logistic regression
    print("\n=== LOGISTIC REGRESSION ===")
    logreg = MyLogisticRegression(X, y)
    logreg.split(test_size=0.20)
    # logreg.cross_validate_penalty(C_range=[0.1, 1, 10, 100, 1000], k_fold_nb=5)
    logreg.init_model(C=10, max_iterations=10000)
    logreg.train()
    # logreg.confusion_matrix()
    logreg.print_report()
    logreg.plot_roc_curve(display_plot=False)

    # (i)(iv) kNN
    print("\n=== K NEAREST NEIGHBORS CLASSIFIER ===")
    knn = MyKNeighborsClassifier(X, y)
    knn.split(test_size=0.20)
    # knn.cross_validate_n(n_range=[1, 5, 10, 50, 100, 500, 1000], k_fold_nb=5)
    # knn.cross_validate_n(n_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k_fold_nb=5)
    knn.init_model(n=3)
    knn.train()
    knn.confusion_matrix()
    knn.print_report()
    knn.plot_roc_curve(display_plot=False)

    # (i)(v) MLP
    print("\n=== MLP CLASSIFIER ===")
    mlp = MyMLPClassifier(X, y)
    mlp.split(test_size=0.20)
    # mlp.cross_validate_n(prev_hidden_layers=(), n_range=[1, 5, 10, 25, 50, 100, 500, 1000], max_iterations=10000)
    mlp.cross_validate_n(prev_hidden_layers=(50,), n_range=[1, 5, 10, 25, 50, 100], max_iterations=10000)
    # mlp.cross_validate_n(prev_hidden_layers=(50, 5), n_range=[1, 5, 10, 25, 50, 100], max_iterations=10000)
    mlp.init_model(n=(50,))
    mlp.train()
    # mlp.save_model("saved-models/MyMLPClassifier-50.joblib")
    # mlp.load_model()
    mlp.confusion_matrix()
    mlp.print_report()
    mlp.plot_roc_curve(display_plot=False)

    # Print all ROC Curves in a single figure
    plt.figure(123) # ROC Curves are added by each model on figure #123
    plt.title("ROC Curves of different classifiers")
    plt.show()


if part_ii:
    # (ii)(i) Preprocessing data
    print("> Preprocessing data")
    preprocessor = MyPreprocessor()
    preprocessor.read_data('dataset/reviews_17.jl.json')
    preprocessor.preprocess_data(predict="early_access", min_df=1, max_df=0.05)
    X, y = preprocessor.get_data()

    preprocessor.print_report()

    # (ii)(ii) Baseline model
    print("\n=== BASELINE MODEL ===")
    baseline = MyBaseline(X, y)
    baseline.split(test_size=0.20)
    baseline.init_model()
    baseline.train()
    # baseline.confusion_matrix()
    baseline.print_report()
    baseline.plot_roc_curve(display_plot=False)

    # (ii)(iii) Logistic regression
    print("\n=== LOGISTIC REGRESSION ===")
    logreg = MyLogisticRegression(X, y)
    logreg.split(test_size=0.20)
    # logreg.cross_validate_penalty(C_range=[0.1, 1, 10, 100, 1000, 5000], k_fold_nb=5)
    logreg.init_model(C=1000, max_iterations=10000)
    logreg.train()
    # logreg.confusion_matrix()
    logreg.print_report()
    logreg.plot_roc_curve(display_plot=False)

    # (ii)(iv) kNN
    print("\n=== K NEAREST NEIGHBORS CLASSIFIER ===")
    knn = MyKNeighborsClassifier(X, y)
    knn.split(test_size=0.20)
    # knn.cross_validate_n(n_range=[1, 5, 10, 50, 100, 500, 1000], k_fold_nb=5)
    # knn.cross_validate_n(n_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], k_fold_nb=5)
    knn.init_model(n=1)
    knn.train()
    # knn.confusion_matrix()
    knn.print_report()
    knn.plot_roc_curve(display_plot=False)

    # (ii)(v) MLP
    print("\n=== MLP CLASSIFIER ===")
    mlp = MyMLPClassifier(X, y)
    mlp.split(test_size=0.20)
    # mlp.cross_validate_n(prev_hidden_layers=(), n_range=[1, 5, 10, 25, 50, 100, 500, 1000], max_iterations=10000)
    mlp.init_model(n=(50,))
    mlp.train()
    # mlp.save_model("saved-models/MyMLPClassifier-ii-50.joblib")
    # mlp.load_model()
    # mlp.confusion_matrix()
    mlp.print_report()
    mlp.plot_roc_curve(display_plot=False)

    # Print all ROC Curves in a single figure
    plt.figure(123)  # ROC Curves are added by each model on figure #123
    plt.title("ROC Curves of different classifiers")
    plt.show()