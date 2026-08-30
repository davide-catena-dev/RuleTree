import numpy as np
import copy
from scipy.stats import gaussian_kde
from sklearn.utils import check_random_state
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification.TrepanStumpClassifier import TrepanStumpClassifier

class MofNTrepanStumpClassifier(TrepanStumpClassifier):
    def __init__(self, max_conditions=3, **kwargs):
        super().__init__(**kwargs)
        self.max_conditions = max_conditions
        self.m = 1
        self.conditions = []
        self.is_categorical = False
        self.left_class = None
        self.right_class = None
        

    def _apply_condition(self, X: np.ndarray, cond: tuple) -> np.ndarray:
        """Helper vettorizzato per valutare una singola condizione su tutti i campioni."""
        feat_idx, thresh, op = cond
        col = X[:, feat_idx]
        
        if op == "<=":
            return col <= thresh
        elif op == "==":
            return col == thresh
        elif op == "!=":
            return col != thresh
        else:
            return col > thresh

    def _evaluate_rule(self, X, y, m, conditions, weights):
        """Calcolo del Gain Ratio Vettorizzato con Caching"""
        if not conditions:
            return float('inf'), np.zeros(X.shape[0], dtype=bool), np.ones(X.shape[0], dtype=bool)
        
        # 1. CREAZIONE CHIAVE DI CACHE
        cache_key = (m, frozenset(conditions))
        
        # Se abbiamo già calcolato questo set di condizioni per questo nodo, restituiscilo!
        if hasattr(self, '_eval_cache') and cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        # VETTORIZZAZIONE: Creiamo una matrice (n_samples, n_conditions) e sommiamo in asse 1
        masks = np.column_stack([self._apply_condition(X, cond) for cond in conditions])
        satisfied_counts = np.sum(masks, axis=1)
        
        left_mask = satisfied_counts >= m
        right_mask = ~left_mask
        
        # Uscita anticipata per tagli inutili
        if not np.any(left_mask) or not np.any(right_mask):
            result = (float('inf'), left_mask, right_mask)
            # SALVATAGGIO IN CACHE (Anche per i casi pessimi)
            if hasattr(self, '_eval_cache'):
                self._eval_cache[cache_key] = result
            return result
        
        def entropy(mask):
            mask_weights = weights[mask]
            total_weight = np.sum(mask_weights)
            if total_weight == 0: 
                return 0.0
            
            # Estrazione classi e probabilità in modo array-based
            classes = np.unique(y[mask])
            probs = np.array([np.sum(mask_weights[y[mask] == c]) for c in classes]) / total_weight
            
            # Filtro probabilità per evitare log2(0)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
            
        n = np.sum(weights)
        weight_left = np.sum(weights[left_mask])
        weight_right = np.sum(weights[right_mask])
        
        # Calcolo Information Gain
        parent_entropy = entropy(np.ones(len(y), dtype=bool))
        entropy_left = entropy(left_mask)
        entropy_right = entropy(right_mask)
        
        ig = parent_entropy - ((weight_left / n) * entropy_left + (weight_right / n) * entropy_right)
        
        # Calcolo Split Info e Gain Ratio
        split_info = -((weight_left / n) * np.log2(weight_left / n) + (weight_right / n) * np.log2(weight_right / n))
        
        gain_ratio = (ig / split_info) if split_info > 1e-9 else 0.0
        
        # Prepariamo il risultato finale
        result = (-gain_ratio, left_mask, right_mask)
        
        # SALVATAGGIO IN CACHE (Prima di restituirlo)
        if hasattr(self, '_eval_cache'):
            self._eval_cache[cache_key] = result
            
        return result

        

    def fit(self, X=None, y=None, X_ts=None, X_img=None, X_txt=None,
            idx=None, context=None, sample_weight=None, check_input=True):
        
        if X is None or y is None: return self
        seed = getattr(context, 'random_state', getattr(self, 'random_state', None))
        rng = check_random_state(seed)
        if idx is None: idx = slice(None)
        
        self._eval_cache = {}
            
        X_local = X[idx]
        y_local = y[idx]
        
        # EARLY EXIT: Se non ci sono dati, non tentiamo lo split
        if len(X_local) == 0:
            raise NoSplitFoundWarning("Nessun dato disponibile per lo split.")

        # 1. ORACLE E ETICHETTATURA LOCALE
        if context is not None and hasattr(context, 'oracle'):
            try: y_labels = np.asarray(context.oracle.predict(X_local)).ravel()
            except Exception: y_labels = y_local
        else:
            y_labels = y_local
            
        S_MIN = getattr(context, 's_min', 1000)
        
        # 2. GENERAZIONE DATI SINTETICI (Tramite il generatore centralizzato)
        if context is not None and hasattr(context, '_current_node') and 0 < len(X_local) < S_MIN:
            n_synth = S_MIN - len(X_local)
            
            constraints = getattr(context._current_node, 'constraints', [])
            m_of_n_rules = getattr(context._current_node, 'm_of_n_rules', [])
            generator = getattr(context, '_synth_gen', None)
            
            if generator is not None:
                # Generazione in una singola riga!
                X_synth = generator.sample_constrained(n_synth, constraints, m_of_n_rules)
                
                # Prevenzione Array Vuoto: Interroghiamo l'oracolo solo se ci sono dati validi
                if len(X_synth) > 0:
                    y_synth = np.asarray(context.oracle.predict(X_synth)).ravel()
                    X_local = np.vstack((X_local, X_synth))
                    y_labels = np.concatenate((y_labels, y_synth))
                else:
                    import warnings
                    warnings.warn("Nessun campione sintetico generato per il nodo. Uso solo dati reali.")

        if np.unique(y_labels).size <= 1:
            raise NoSplitFoundWarning(f"Nodo puro: impossibile splittare su y {np.unique(y_labels)}")

        weights_og = sample_weight[idx] if sample_weight is not None else None
        if weights_og is not None:
            n_synth = len(y_labels) - len(weights_og)
            if n_synth > 0:
                weights_og = np.concatenate((weights_og, np.ones(n_synth) * np.mean(weights_og)))
            local_weights = weights_og
        else:
            local_weights = np.ones(len(y_labels))

        # 3. RICERCA DEL SEME
        n_samples, n_features = X_local.shape
        best_seed_score = float('inf')
        best_seed = None
        
        for feat_idx in range(n_features):
            uniq_vals = np.unique(X_local[:, feat_idx])
            is_categorical = len(uniq_vals) <= 10  
            
            if is_categorical:
                thresholds = uniq_vals
                operators = ["=="]  
            else:
                thresholds = uniq_vals
                if len(thresholds) > 100: 
                    thresholds = np.percentile(X_local[:, feat_idx], np.linspace(1, 99, 100))
                operators = ["<="]
                
            for thresh in thresholds:
                for op in operators:
                    score, _, _ = self._evaluate_rule(X_local, y_labels, m=1, conditions=[(feat_idx, thresh, op)], weights=local_weights)
                    if score < best_seed_score:
                        best_seed_score = score
                        best_seed = (feat_idx, thresh, op)
                        
        if best_seed is None or best_seed_score == float('inf'):
            raise NoSplitFoundWarning("Nessun taglio valido trovato.")
                    
        self.m = 1
        self.conditions = [best_seed]
        current_score = best_seed_score
        used_features = {best_seed[0]}
        
        if context is not None and hasattr(context, '_current_node') and context._current_node is not None:
            for c in getattr(context._current_node, 'constraints', []):
                used_features.add(c.feature_index)
        
        # 4. ESPANSIONE M-OF-N 
        improved = True
        while improved and len(self.conditions) < self.max_conditions:
            improved = False
            best_cand_score = current_score
            best_cand_m = self.m
            best_cand_cond = None
            
            for feat_idx in range(n_features):
                if feat_idx in used_features: continue
                
                uniq_vals = np.unique(X_local[:, feat_idx])
                is_categorical = len(uniq_vals) <= 10
                
                if is_categorical:
                    thresholds = uniq_vals
                    operators = ["==", "!="]  
                else:
                    thresholds = uniq_vals
                    if len(thresholds) > 20: 
                        thresholds = np.percentile(X_local[:, feat_idx], np.linspace(5, 95, 20))
                    operators = ["<=", ">"]
                    
                for thresh in thresholds:
                    for op in operators: 
                        new_cond = (feat_idx, thresh, op)
                        cand_conditions = self.conditions + [new_cond]
                        
                        score_op1, _, _ = self._evaluate_rule(X_local, y_labels, self.m, cand_conditions, local_weights)
                        score_op2, _, _ = self._evaluate_rule(X_local, y_labels, self.m + 1, cand_conditions, local_weights)
                        
                        if score_op1 < (best_cand_score - 1e-6):
                            best_cand_score = score_op1
                            best_cand_m = self.m
                            best_cand_cond = new_cond
                            
                        if score_op2 < (best_cand_score - 1e-6):
                            best_cand_score = score_op2
                            best_cand_m = self.m + 1
                            best_cand_cond = new_cond
                            
            if best_cand_cond is not None:
                self.m = best_cand_m
                self.conditions.append(best_cand_cond)
                used_features.add(best_cand_cond[0])
                current_score = best_cand_score
                improved = True
                
        # 5. CHIUSURA E ROUTING
        _, left_mask, right_mask = self._evaluate_rule(X_local, y_labels, self.m, self.conditions, local_weights)
        
        # Sganciamo la predizione reale dal routing logico dell'albero RuleTree
        self.left_class = 1   # ID Ramo Sinistro
        self.right_class = 2  # ID Ramo Destro
        
        if weights_og is not None:
            n_parent = np.sum(local_weights)
            n_left = np.sum(local_weights[left_mask])
            n_right = np.sum(local_weights[right_mask])
        else:
            n_parent = float(len(y_labels))
            n_left = float(np.sum(left_mask))
            n_right = float(np.sum(right_mask))

        self.feature_original = np.array([self.conditions[0][0], -2, -2]) if self.conditions else np.array([-2, -2, -2])
        self.threshold_original = np.array([self.conditions[0][1], -2, -2]) if self.conditions else np.array([-2, -2, -2])
        self.impurity = [current_score, 0.0, 0.0]
        
        class DummyTree:
            pass
        self.tree_ = DummyTree()
        self.tree_.impurity = self.impurity
        self.tree_.feature = self.feature_original
        self.tree_.threshold = self.threshold_original
        self.tree_.weighted_n_node_samples = [n_parent, n_left, n_right]
        self.tree_.n_node_samples = [len(y_labels), np.sum(left_mask), np.sum(right_mask)]
        
        # FIX Categoria per formattazione testuale regole
        if self.conditions:
            self.is_categorical = (self.conditions[0][2] in ["==", "!="])
        
        return self

    def apply(self, X=None, X_ts=None, X_img=None, X_txt=None, idx=None, check_input=False):
        if idx is not None: 
            X = X[idx]
            
        satisfied_counts = np.zeros(X.shape[0])
        for feat_idx, thresh, op in self.conditions:
            if op == "<=": satisfied_counts += (X[:, feat_idx] <= thresh).astype(int)
            elif op == "==": satisfied_counts += (X[:, feat_idx] == thresh).astype(int)
            elif op == "!=": satisfied_counts += (X[:, feat_idx] != thresh).astype(int)
            else: satisfied_counts += (X[:, feat_idx] > thresh).astype(int)
                
        # Creiamo un array di 2 (Ramo Destro / Condizione Falsa)
        y_pred = np.ones(X.shape[0]) * 2
        
        # Sovrascriviamo con 1 (Ramo Sinistro / Condizione Vera) dove la regola M-of-N è rispettata
        y_pred[satisfied_counts >= self.m] = 1
        
        return y_pred
    
    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        cond_strings = []
        for feat_idx, thresh, op in self.conditions:
            feat_name = f"X_{feat_idx}" if columns_names is None else columns_names[feat_idx]
            
            if scaler is not None:
                array = np.zeros((1, scaler.n_features_in_))
                array[0, feat_idx] = thresh
                thresh = scaler.inverse_transform(array)[0, feat_idx]
                
            rounded_thresh = round(thresh, float_precision) if float_precision is not None else thresh
            cond_strings.append(f"{feat_name} {op} {rounded_thresh}")
            
        rule_text = f"{self.m}-of-{{ " + ", ".join(cond_strings) + " }}"
        
        rule = {
            "feature_idx": self.conditions[0][0] if self.conditions else -2,
            "threshold": self.conditions[0][1] if self.conditions else -2,
            "is_categorical": self.is_categorical,
            "feature_name": "Multiple",
            "textual_rule": rule_text,
            "blob_rule": rule_text,
            "graphviz_rule": {"label": rule_text},
            "not_textual_rule": f"NOT ({rule_text})",
            "not_blob_rule": f"NOT ({rule_text})",
            "not_graphviz_rule": {"label": f"NOT ({rule_text})"}
        }
        return rule

    def node_to_dict(self):
        rule = super().node_to_dict()
        rule["m_threshold"] = self.m
        rule["conditions"] = self.conditions
        rule["left_class"] = self.left_class
        rule["right_class"] = self.right_class
        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        self = super().dict_to_node(node_dict, X)
        self.m = node_dict.get("m_threshold", 1)
        self.conditions = node_dict.get("conditions", [])
        self.left_class = node_dict.get("left_class", 0)
        self.right_class = node_dict.get("right_class", 0)
        return self
        