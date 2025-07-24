"""
Vlastná obalová trieda klasifikátora pre tradičné algoritmy (Random Forest, KNN, Decision Tree)

Tento modul definuje triedu `CustomClassifier`, ktorá slúži ako obal nad tradičné klasifikačné algoritmy
zo scikit-learn. Umožňuje flexibilne zvoliť konkrétny typ modelu a jeho hyperparametre. Implementácia
je kompatibilná so scikit-learn API (fit/predict) a podporuje modely: RandomForest, k-Nearest Neighbors (KNN),
a Decision Tree. Tréning jednotlivých modelov využíva pomocné funkcie zo súboru `traditional_model.py`.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from src.classical.traditional_model import train_random_forest, train_knn, \
    train_decision_tree
class CustomClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_type='RandomForest', criterion=None, max_depth=None, max_features=None,
                 min_samples_leaf=1, min_samples_split=2, n_estimators=100,
                 C=1.0, degree=3, gamma='scale', kernel='rbf',
                 n_neighbors=5, metric='minkowski', weights='uniform'):
        """
                Inicializačná metóda – nastavuje typ modelu a jeho hyperparametre.

                Parametre:
                - model_type: Typ modelu ('RandomForest', 'KNN', 'DecisionTree')
                - criterion, max_depth, atď.: Hyperparametre pre jednotlivé modely
                """
        self.model_type = model_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.model = None

    def fit(self, X, y):
        """
                Tréning modelu na vstupe X (črty) a y (triedy).
                Vybraný model sa trénuje podľa nastaveného typu `model_type`.
                """
        if self.model_type == 'RandomForest':
            self.model = train_random_forest(
                X, y,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                n_estimators=self.n_estimators
            )
        elif self.model_type == 'KNN':
            self.model = train_knn(
                X, y,
                metric=self.metric,
                n_neighbors=self.n_neighbors,
                weights=self.weights
            )
        elif self.model_type == 'DecisionTree':
            self.model = train_decision_tree(
                X, y,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split
            )
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")
        return self

    def predict(self, X):
        """
               Predikcia výstupných tried pre vstupné dáta X pomocou natrénovaného modelu.
               """
        return self.model.predict(X)
