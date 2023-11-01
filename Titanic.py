from catboost.datasets import titanic
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
# Титаник деректер жиынтығынан деректерді жүктеу
train_df, test_df = titanic()
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)
# Белгілер мен мақсатты айнымалыны анықтау
X = train_df.drop('Survived', axis=1)
y = train_df.Survived
# Категориялық белгілерді анықтау
categorical_features_indices = np.where(X.dtypes != float)[0]
# Деректерді жаттығу (тренеровочные) және валидация жиынтығына бөлу
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
# Catboostclassifier нысанын құру
model = CatBoostClassifier(iterations=500,  # Оқытуға арналған итерациялар (ағаштар) саны
                           learning_rate=0.1,  # Оқу қадамы
                           depth=6,  # Ағаштардың тереңдігі
                           loss_function='Logloss',  # Жіктеуге арналған шығындар функциясы
                           cat_features = categorical_features_indices  # Категориялық белгілер индекстері
                        )
# Жаттығу деректері үшін Pool нысанын құру
train_data = Pool(data=X_train, label=y_train, cat_features= categorical_features_indices)
# Модельді оқыту
model.fit(train_data, eval_set=(X_validation, y_validation))
# Сынақ деректеріндегі болжамдар
X_test = test_df
# Айнымалыдағы нәтижелер 'predictions'
predictions = model.predict(X_test)

