import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from RuleTree.tree.TrepanClassifier import Oracle, TrepanClassifier, TrepanNode
from RuleTree.stumps.classification.MofNTrepanStumpClassifier import MofNTrepanStumpClassifier
import matplotlib

from RuleTree.utils.feature_utils import detect_categorical_features
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#from RuleTree.tree.TrepanClassifier import Oracle
from sklearn.preprocessing import LabelEncoder
import utils as u
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
df = pd.read_csv("datasets/CLF/adult.csv", skipinitialspace=True, na_values="?")

# 2. Controllo di sicurezza: rimozione di eventuali spazi residui nei nomi delle colonne
df.columns = df.columns.str.strip()

# 3. Ora la tua riga di codice funzionerà senza errori!
# Sostituiamo i NaN con la stringa 'Missing' per la colonna workclass (o all'interno del tuo ciclo)
df["workclass"] = df["workclass"].fillna("Missing")

# Se vuoi applicare la sostituzione a tutte le colonne di testo con valori mancanti:
colonne_con_mancanti = ["workclass", "occupation", "native-country"]
for m in colonne_con_mancanti:
    df[m] = df[m].fillna("Missing")


le = LabelEncoder()
binary = ['class','sex']
df['class'] = df['class'].str.replace('.','',regex=False)
for a in binary:
    df[a] = le.fit_transform(df[a])
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].astype(str) # Assicurati siano tutte stringhe
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
df.head()

attributes = [col for col in df.columns if col != 'class']
X = df[attributes].values
y = df['class']
categorical_features = detect_categorical_features(df[attributes])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.utils.class_weight import compute_sample_weight
# Calcola un array lungo quanto y_train, dove la classe 1 avrà un valore 
# numerico più alto della classe 0 per compensare lo sbilanciamento.
pesi_bilanciati = compute_sample_weight(class_weight='balanced', y=y_train)

rd = RandomForestClassifier(n_estimators=200, random_state=42)
rd.fit(X_train, y_train)
y_test_pred = rd.predict(X_test)
y_train_pred = rd.predict(X_train)

print('Train Accuracy %s' % accuracy_score(y_train, y_train_pred))
print('Train F1-score %s' % f1_score(y_train, y_train_pred, average=None))
print()

print('Test Accuracy %s' % accuracy_score(y_test, y_test_pred))
print('Test F1-score %s' % f1_score(y_test, y_test_pred, average=None))

print(classification_report(y_test, y_test_pred))

tr = TrepanClassifier(estimator=rd,random_state=42, s_min=1000, categorical_features=categorical_features, max_internal_nodes=15)
tr.fit(X_train, y_train, sample_weight=pesi_bilanciati)
y_pred = tr.predict(X_test)



print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_test_oracle = rd.predict(X_test)
fidelity = accuracy_score(y_test_oracle, y_pred)
print("Fedeltà di Trepan rispetto all'Oracolo: %.2f%%" % (fidelity * 100))

tr.compute_feature_importances()

rules = tr.get_rules(columns_names=df.columns)
# Print textual rules of trained tree
#tr.print_rules(rules)

feature_importances = tr.compute_feature_importances()
feature_names = attributes
#print(feature_importances)

# 1. Assicuriamoci che l'array delle importanze sia lungo quanto i nomi delle feature
if len(feature_importances) < len(feature_names):
    padded_importances = np.zeros(len(feature_names))
    padded_importances[:len(feature_importances)] = feature_importances
    feature_importances = padded_importances

# 2. Ora la creazione del DataFrame Pandas filmerà liscia senza errori
feat_importances = pd.DataFrame(feature_importances, index=feature_names, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.head(5).plot(kind='bar', figsize=(10,6), color = '#002d9c')
plt.xticks(fontsize=20, rotation = 45)
plt.yticks(fontsize=20)                       
plt.legend(fontsize=20)  
plt.show()

prediction, bias, contributions = tr.local_interpretation(X_test, joint_contribution= True)
tr.print_trepan_rules(feature_names=attributes)
print('First instance analysis')
print(f'Prediction: {prediction[0]}')
print(f'Bias: {bias[0]}')
print(f'Contributions: {contributions[0]}')



trm = TrepanClassifier(estimator=rd, max_leaf_nodes=20, random_state=42, categorical_features=categorical_features)
trm.fit(X_train, y_train, sample_weight=pesi_bilanciati)
y_pred = trm.predict(X_test)



print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_test_oracle = rd.predict(X_test)
fidelity = accuracy_score(y_test_oracle, y_pred)
print("Fedeltà di Trepan rispetto all'Oracolo: %.2f%%" % (fidelity * 100))

trm.compute_feature_importances()

rules = trm.get_rules(columns_names=df.columns)
# Print textual rules of trained tree
#tr.print_rules(rules)

feature_importances = trm.compute_feature_importances()
feature_names = attributes
#print(feature_importances)

# 1. Assicuriamoci che l'array delle importanze sia lungo quanto i nomi delle feature
if len(feature_importances) < len(feature_names):
    padded_importances = np.zeros(len(feature_names))
    padded_importances[:len(feature_importances)] = feature_importances
    feature_importances = padded_importances

# 2. Ora la creazione del DataFrame Pandas filmerà liscia senza errori
feat_importances = pd.DataFrame(feature_importances, index=feature_names, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.head(5).plot(kind='bar', figsize=(10,6), color = '#002d9c')
plt.xticks(fontsize=20, rotation = 45)
plt.yticks(fontsize=20)                       
plt.legend(fontsize=20)  
plt.show()

prediction, bias, contributions = trm.local_interpretation(X_test, joint_contribution= True)
trm.print_trepan_rules(feature_names=attributes)
print('First instance analysis')
print(f'Prediction: {prediction[0]}')
print(f'Bias: {bias[0]}')
print(f'Contributions: {contributions[0]}')


# 1. Definisci lo spazio di ricerca (i parametri della classe TrepanClassifier)
param_grid = {
    'max_depth': [3,4,6,8,None],
    'max_leaf_nodes': [10,20,30, 50, None],
    's_min': [1000, 2000, 3000]
}

# 2. Inizializza il classificatore base
# Passiamo una callable che invoca il RandomForest già addestrato `rd`.
# In questo modo `sklearn.clone` non rimuove lo stato fitted e le copie
# usate da GridSearch chiameranno comunque il `predict` del modello addestrato.
tr_base = TrepanClassifier(estimator=(lambda X, _rd=rd: _rd.predict(X)), categorical_features=categorical_features, random_state=42)

def refit_complexity(results):
    scores = results["mean_test_score"]
    params = results["params"]

    def normalize_complexity(param_value):
        return 1000 if param_value is None else param_value

    best_index = None
    best_value = -float("inf")
    for i, (score, p) in enumerate(zip(scores, params)):
        complexity = normalize_complexity(p["max_leaf_nodes"]) + normalize_complexity(p["max_depth"])
        value = score - 0.001 * complexity
        if value > best_value:
            best_value = value
            best_index = i
    return best_index

def refit_simple(results):
    scores = results["mean_test_score"]
    params = results["params"]

    def normalize_complexity(param_value):
        return 1000 if param_value is None else param_value

    best_score = max(scores)
    best_indices = [i for i, s in enumerate(scores) if s >= best_score - 1e-6]

    best_index = min(
        best_indices,
        key=lambda i: normalize_complexity(params[i]["max_leaf_nodes"]) +
                      normalize_complexity(params[i]["max_depth"])
    )
    return best_index

# 3. Configura la GridSearchCV
grid_search = GridSearchCV(
    estimator=tr_base,
    param_grid=param_grid,
    scoring='accuracy',  # Userà l'accuratezza, ma truccata!
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    refit=refit_simple,
    n_jobs=-1,           # Usa tutti i core del PC per fare in fretta
    verbose=2
)

# 4. IL TRUCCO: Generiamo le "etichette oracolo" per l'addestramento
oracle_y_train = rd.predict(X_train)

# 5. Addestriamo la GridSearch puntando alla FEDELTÀ
print("Avvio GridSearch per massimizzare la Fedeltà...")
grid_search.fit(X_train, oracle_y_train, sample_weight=pesi_bilanciati)

# 6. Risultati
print("\n🏆 MIGLIORI PARAMETRI TROVATI:", grid_search.best_params_)
best_score = grid_search.cv_results_["mean_test_score"][grid_search.best_index_]
print(f"🏆 FEDELTÀ STIMATA (CV) DEL PARAMETRO SELEZIONATO: {best_score * 100:.2f}%")

# Il modello già ottimizzato e pronto da stampare
best_trepan = grid_search.best_estimator_

y_test_oracle = rd.predict(X_test)
best_pred = best_trepan.predict(X_test)
fidelity_test = accuracy_score(y_test_oracle, best_pred)
#best_trepan.print_trepan_rules(feature_names=attributes)

y_pred = best_trepan.predict(X_test)



print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
print("Fidelity finale su test: %.2f%%" % (fidelity_test * 100))