import numpy as np

# Assicurati che questo import punti correttamente al file che mi hai appena mandato
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.exceptions import NoSplitFoundWarning
from scipy.stats import gaussian_kde
from sklearn.utils import check_random_state

class TrepanStumpClassifier(DecisionTreeStumpClassifier):
    """
    Stump specializzato per Trepan.
    Eredita tutta la logica di split, estrazione regole e serializzazione da DecisionTreeStumpClassifier.
    Sovrascrive solo il metodo fit per usare le etichette dell'Oracolo (e generare dati sintetici).
    """
    
    def fit(self, X=None, y=None, X_ts=None, X_img=None, X_txt=None,
            idx=None, context=None, sample_weight=None, check_input=True):
        
        if X is None or y is None:
            return self
        
        seed = getattr(context, 'random_state', getattr(self, 'random_state', None))
        rng = check_random_state(seed)

        # 1. Isoliamo i dati locali (usando lo stesso approccio della classe madre)
        if idx is None:
            idx = slice(None)
            
        X_local = X[idx]
        y_local = y[idx]

        # 2. LA MAGIA DI TREPAN: Etichettatura con Oracolo
        if context is not None and hasattr(context, 'oracle'):
            try:
                y_labels = np.asarray(context.oracle.predict(X_local)).ravel()
            except Exception:
                y_labels = y_local
        else:
            y_labels = y_local
            
            
            
        local_weights = sample_weight[idx] if sample_weight is not None else None
        
        S_MIN = getattr(context, 's_min', 1000)  # Soglia minima di campioni desiderata
        
        if context is not None and hasattr(context, '_current_node') and 0 < len(X_local) < S_MIN:
            n_synth = S_MIN - len(X_local)
            n_features = X_local.shape[1]
            # Recuperiamo i vincoli spaziali dal nodo corrente (il Bounding Box)
            constraints = context._current_node.constraints if hasattr(context, '_current_node') else [] 
            # Recuperiamo l'intero dataset per conoscere i limiti minimi e massimi assoluti
            global_x = getattr(context, '_X_train_temp', X_local)
            feat_mins = np.min(global_x, axis=0).astype(float)
            feat_maxs = np.max(global_x, axis=0).astype(float)
            # Restringiamo i limiti globali applicando i Constraints del percorso logico
            for c in constraints:
                f_idx = c.feature_index
                val = float(c.value)
                if c.operator == "<=":
                    feat_maxs[f_idx] = min(feat_maxs[f_idx], val)
                elif c.operator == ">=":
                    feat_mins[f_idx] = max(feat_mins[f_idx], val)
                    
            # Matrice vuota per i nuovi dati sintetici
            X_synth = np.zeros((n_synth, n_features))
            
            for f in range(n_features):
                col_data = global_x[:, f]
                uniq, counts = np.unique(col_data, return_counts=True)
                
                if len(uniq) <= 10:
                    probs = counts / counts.sum()
                    valid_indices = np.where((feat_mins[f] <= uniq) & (uniq <= feat_maxs[f]))[0]
                    valids_val = uniq[valid_indices]
                    valid_probs = probs[valid_indices]
                    if len(valids_val) > 0:
                        valid_probs = valid_probs / valid_probs.sum()  # Normalizziamo le probabilità
                        X_synth[:, f] = rng.choice(valids_val, size=n_synth, p=valid_probs)
                    else:
                        X_synth[:, f] = uniq[0]
                        
                    
                # SE LA FEATURE È CONTINUA (Usa Kernel Density Estimation - KDE)   
                
                else:
                    # Applichiamo i vincoli ai dati globali prima di stimare la densità
                    valid_data = col_data[(col_data >= feat_mins[f]) & (col_data <= feat_maxs[f])]
                    if len(valid_data) > 1:
                        try:
                            kde = gaussian_kde(valid_data)
                            
                            sampled_vals = kde.resample(n_synth, seed=rng)[0]
                            sampled_vals = np.clip(sampled_vals, feat_mins[f], feat_maxs[f])
                            X_synth[:, f] = sampled_vals
                            
                        # 2. FIX ROBUSTEZZA: Catturiamo solo l'errore specifico di KDE (matrice singolare)
                        except np.linalg.LinAlgError:
                            X_synth[:, f] = rng.uniform(feat_mins[f], feat_maxs[f], size=n_synth)
                    else:
                        X_synth[:, f] = rng.uniform(feat_mins[f], feat_maxs[f], size=n_synth)
                    
            # 3. FIX ROBUSTEZZA: Niente più 'try/except: pass' silenti sull'Oracolo.
            # Se l'Oracolo fallisce, è un errore critico architetturale che DEVE essere sollevato,
            # altrimenti rischiamo di addestrare un albero fallato senza saperlo.
            y_synth = np.asarray(context.oracle.predict(X_synth)).ravel()
            
            # Fusione dei dati sintetici con quelli reali
            X_local = np.vstack((X_local, X_synth))
            y_labels = np.concatenate((y_labels, y_synth))
            
            if local_weights is not None:
                local_weights = np.concatenate((local_weights, np.ones(n_synth)))

        # 3. FRENO A MANO ANTI-LOOP
        # Se l'Oracolo ha predetto una sola classe per tutti i campioni locali,
        # lo split è inutile. Alziamo subito l'eccezione come fa la classe madre.
        if np.unique(y_labels).size <= 1:
            raise NoSplitFoundWarning(f"Nodo puro: impossibile splittare su y {np.unique(y_labels)}")
        
        

        # 4. Deleghiamo il calcolo matematico dello split alla classe madre!
        # ATTENZIONE: Passiamo X_local e y_labels, ma impostiamo idx=None 
        # perché i dati li abbiamo già tagliati noi al punto 1.
        return super().fit(X=X_local, 
                           y=y_labels, 
                           X_ts=X_ts, 
                           X_img=X_img, 
                           X_txt=X_txt,
                           idx=None,  # Fondamentale per non affettare due volte!
                           context=context, 
                           sample_weight=local_weights,
                           check_input=check_input)