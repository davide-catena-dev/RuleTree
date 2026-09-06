import heapq
import numpy as np
from typing import Any, Optional, Union
import torch 
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.tree.TrepanNode import TrepanNode, Constraint
from RuleTree.stumps.classification.MofNTrepanStumpClassifier import MofNTrepanStumpClassifier
import scipy.stats
from RuleTree.utils.synthetic_data import SyntheticDataGenerator
from RuleTree.utils.feature_utils import detect_categorical_features
from typing import Any, Optional
import itertools

class Oracle:
    """Wrapper per interrogare diversi tipi di stimatori/oggetti.
    - predict_proba fallback -> argmax
    - normalizza output in array NumPy 1D
    """
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def _to_numpy(self, arr):
        # Gestisce tensori torch/tf ecc.
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        elif hasattr(arr, "numpy"):
            arr = arr.numpy()
        return np.asarray(arr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Adesso, se l'estimator era un modello PyTorch, qui dentro
        # self.estimator sarà l'InternalPyTorchWrapper, non il modello puro!
        # Quindi entrerà in questo primo 'if', trovando il metodo 'predict'

        if hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X)
        elif hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            probs = self._to_numpy(probs)
            preds = np.argmax(probs, axis=1)
        elif callable(self.estimator):
            preds = self.estimator(X)
        else:
            raise TypeError("Estimator has no predict/predict_proba and is not callable")

        preds = self._to_numpy(preds).ravel()
        return preds

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            return self._to_numpy(probs)
        return None


class InternalPyTorchWrapper:
    """Adapter interno per rendere compatibili i modelli PyTorch con l'Oracle di RuleTree."""
    def __init__(self, model):
        self.model = model
        self.model.eval() 
        self.device = next(self.model.parameters()).device

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


class TrepanClassifier(RuleTreeClassifier):
    def __init__(self,
                 estimator,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps = None,
                 stump_selection: str = 'best',
                 random_state=None,
                 distance_measure=None,
                 s_min: int =1000,
                 epsilon : float = 0.05,
                 delta : float = 0.05,
                 categorical_features=None,       
                 categorical_threshold: int = 10,
                 min_real_samples: int = 30,
                 max_internal_nodes=float('inf')):
        # se non inserisco altri stump, uso MofNTrepanStumpClassifier di default
        if base_stumps is None:
            base_stumps = [MofNTrepanStumpClassifier()]
        # Se per caso passo uno stump singolo, lo inserisco in una lista
        elif not isinstance(base_stumps, list):
            base_stumps = [base_stumps]

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state,
                         distance_measure=distance_measure)
        if estimator is None:
            raise ValueError("estimator must be provided to TrepanClassifier")
        
      
        # Se l'estimator ha l'attributo 'parameters' tipico di nn.Module, 
        # lo avvolgiamo automaticamente per farlo dialogare con Scikit-Learn

        if hasattr(estimator, 'parameters'):
            estimator = InternalPyTorchWrapper(estimator)
            
        self.oracle = Oracle(estimator)
        self.estimator = estimator  # Questo ora punterà al wrapper se era un modello PyTorch
        self.s_min = s_min
        self.epsilon = epsilon
        self.delta = delta
        self.categorical_features = categorical_features
        self.categorical_threshold = categorical_threshold
        self.min_real_samples = min_real_samples
        self.max_internal_nodes = max_internal_nodes
        self._internal_nodes_count = 0

    def fit(self, X: np.ndarray = None, y: np.ndarray = None,
            X_ts=None, X_img=None, X_txt=None, sample_weight=None, **kwargs):

        # 1. BLINDARE I DATI: Convertire tutto in array NumPy 
        if X is not None:
            X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)

         # Determinazione delle feature categoriche
        if self.categorical_features is not None:
           
           self.categorical = list(self.categorical_features)
        else:
           # Rilevamento automatico
           
           self.categorical= detect_categorical_features(X, threshold=self.categorical_threshold)

        # Calcola numerical come complemento
        self.numerical = [i for i in range(X.shape[1]) if i not in self.categorical]
        # Mantiene un riferimento temporaneo durante l'esecuzione del fit di base
        self._X_train_temp = X
        self._total_samples = float(self._X_train_temp.shape[0]) if self._X_train_temp is not None else 1.0
        

        # Call base fit
        super().fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight, **kwargs)

        #Pulizia delle variabili temporanee
        self._X_train_temp = None
        return self
    
    
    
        
    def prepare_node(self,
                     y: np.ndarray,
                     idx: np.ndarray,
                     node_id: str,
                     node: Optional[TrepanNode] = None) -> TrepanNode:
        
        # 1) PREVENZIONE CRASH DA DATI SINTETICI
        if idx is None or len(idx) == 0:
            parent_node = self._get_parent_dynamically(node_id)
            pred = getattr(parent_node, 'prediction', 0) if parent_node else 0
            
            class DummyBase: pass
            base_node = DummyBase()
            base_node.node_id = node_id
            base_node.prediction = pred
            base_node.prediction_probability = getattr(parent_node, 'prediction_probability', np.array([1.0])) if parent_node else np.array([1.0])
            base_node.log_odds = getattr(parent_node, 'log_odds', np.array([0.0])) if parent_node else np.array([0.0])
            base_node.classes = getattr(parent_node, 'classes', np.unique(y)) if parent_node else np.unique(y)
            setattr(base_node, "parent", parent_node)
        else:
            if getattr(self, '_X_train_temp', None) is None: 
               raise RuntimeError("_X_train_temp non disponibile; chiamare fit() prima di prepare_node?")
        
            X_local = self._X_train_temp[idx] 
            try:
                y_oracle_local = np.asarray(self.oracle.predict(X_local)).ravel()
                y_for_base = y.copy()
                y_for_base[idx] = y_oracle_local
            except Exception as e:
                import warnings
                warnings.warn(f"Oracle predict fallito in prepare_node: {e}. Uso etichette originali.")
                y_for_base = y
            base_node = super().prepare_node(y_for_base, idx, node_id, node)

        # 2) CERTEZZA GEOMETRICA DEI CONSTRAINTS
        node_constraints: list[Constraint] = []
        node_m_of_n_rules = []

        if node_id is not None and len(str(node_id)) > 1:
            parent_node = self._get_parent_dynamically(node_id)
            if parent_node is not None and isinstance(parent_node, TrepanNode):
                node_constraints = parent_node.copy_constraints()
                node_m_of_n_rules = list(getattr(parent_node, 'm_of_n_rules', []))

                padre_stump = getattr(parent_node, "stump", None)
                if padre_stump is not None and hasattr(padre_stump, 'conditions') and len(padre_stump.conditions) > 0:
                    conditions = padre_stump.conditions
                
                    is_left = str(node_id).endswith('l')
                    
                    if isinstance(padre_stump, MofNTrepanStumpClassifier):
                        # Aggiungi sempre le condizioni, indipendentemente dal numero
                        node_m_of_n_rules.append((padre_stump.m, padre_stump.conditions, is_left))
                    elif len(conditions) == 1:
                        feat_idx, thresh, op = conditions[0]
                        feat_idx = int(feat_idx)
                        thresh = float(thresh) 
                        
                        if is_left:
                            node_constraints.append(Constraint(feature_index=feat_idx, operator=op, value=thresh))
                        else:
                            inverse_op = {
                                "<=": ">", ">": "<=", "==": "!=", "!=": "==", "<": ">=", ">=": "<"
                            }.get(op, "!=")
                            node_constraints.append(Constraint(feature_index=feat_idx, operator=inverse_op, value=thresh))

        # 3) CALCOLO FIDELITY E REACH 
        base_pred = base_node.prediction
        is_pure = False

        #  Estrae prediction del nodo (base_label) 
        if np.ndim(base_pred) > 0:
            base_arr = np.asarray(base_pred)
            if base_arr.ndim == 1 and base_arr.size == len(base_arr):
                base_label = int(np.argmax(base_arr)) if (np.all(base_arr >= 0) and np.isclose(base_arr.sum(), 1.0)) else base_arr.ravel()[0]
            else:
                base_label = base_arr.ravel()[0]
        else:
            base_label = base_pred

        # Funzione helper per calcolare fidelity e purezza
        def get_fidelity_and_purity(y_real):
            if len(y_real) == 0:
                return 1.0, True
            matches = np.sum(y_real == base_label)
            fid = matches / len(y_real)
            pure = (len(np.unique(y_real)) == 1) or getattr(self, '_is_statistically_pure', lambda y, e, d: False)(y_real, self.epsilon, self.delta)
            return fid, pure

        total_samples = self._X_train_temp.shape[0] if self._X_train_temp is not None else 1.0

        # 1. REACH: basato su dati reali
        if idx is not None and len(idx) > 0:
            reach = len(idx) / total_samples
        else:
            reach = 0.0

        # 2. FIDELITY
        fidelity = 0.0
        is_pure = False

        # Caso 1: Dati reali sufficienti
        if idx is not None and (len(idx) >= self.min_real_samples):
            X_local_real = self._X_train_temp[idx]
            y_oracle_real = np.asarray(self.oracle.predict(X_local_real)).ravel()
            fidelity, is_pure = get_fidelity_and_purity(y_oracle_real)
            

        # Caso 2: Pochi dati reali ma generatore disponibile
        elif idx is not None and len(idx) > 0:
            X_local = self._X_train_temp[idx]
            
            local_gen = SyntheticDataGenerator(X_local, random_state=self.random_state,
                                       categorical_features=self.categorical)
            
            # Calcolo dinamico basato su s_min e reach
            safe_idx = idx if idx is not None else []
            n_init, n_step, n_max = self._compute_sampling_params(safe_idx, reach, node_constraints, node_m_of_n_rules)
            
            X_constrained = local_gen.sample_constrained(n_init, node_constraints, node_m_of_n_rules)
            
 
            if len(X_constrained) == 0:
                # Fallback: usa i pochi dati reali che abbiamo
                X_local_real = self._X_train_temp[idx]
                y_oracle_real = np.asarray(self.oracle.predict(X_local_real)).ravel()
                fidelity, is_pure = get_fidelity_and_purity(y_oracle_real)
            else:
                # Calcolo su sintetici con espansione iterativa
                y_oracle_constrained = np.asarray(self.oracle.predict(X_constrained)).ravel()
                matches = np.sum(y_oracle_constrained == base_label)

                while len(X_constrained) < n_max:
                    if self._check_leaf_dominance(y_oracle_constrained):
                        break

                    X_extra = local_gen.sample_constrained(n_step, node_constraints, node_m_of_n_rules)
                    if len(X_extra) == 0:
                        break

                    y_extra = np.asarray(self.oracle.predict(X_extra)).ravel()
                    y_oracle_constrained = np.concatenate((y_oracle_constrained, y_extra))
                    matches += np.sum(y_extra == base_label)
                    X_constrained = np.vstack((X_constrained, X_extra))

                fidelity = matches / len(X_constrained)
                _, is_pure = get_fidelity_and_purity(y_oracle_constrained)
        else:
            fidelity = 1.0
            is_pure = True
        # 4) Ritorna il nodo e inietta il flag
        new_node = TrepanNode(
            node_id=base_node.node_id,
            prediction=base_node.prediction,
            prediction_probability=base_node.prediction_probability,
            log_odds=base_node.log_odds,
            classes=base_node.classes,
            parent=getattr(base_node, "parent", None),
            reach=reach,
            fidelity=fidelity,
            stump=getattr(base_node, "stump", None),
            node_l=getattr(base_node, "node_l", None),
            node_r=getattr(base_node, "node_r", None),
            constraints=node_constraints,
            m_of_n_rules=node_m_of_n_rules,
            is_statistically_pure=is_pure
        )
        
        
        return new_node
    
        
    def _get_parent_dynamically(self, node_id: str):
        """
        Naviga l'albero dalla radice per trovare il padre in modo dinamico, 
        evitando il bug della mappa statica del framework base.
        """
        if not node_id or len(str(node_id)) <= 1:
            return None
            
        current = getattr(self, 'root', None)
        if current is None:
            return None
            
        # Salta la 'R' iniziale e naviga 'l' o 'r' fino al padre
        for char in str(node_id)[1:-1]:
            if char == 'l':
                current = getattr(current, 'node_l', None)
            elif char == 'r':
                current = getattr(current, 'node_r', None)
            if current is None:
                return None
        return current
    

    def _is_statistically_pure(self, labels, epsilon=None, delta=None):
        """
        Implementa il criterio di arresto statistico descritto nel paper TREPAN.
        Valuta se prob(p_c < (1 - epsilon)) < delta usando l'approssimazione normale.
        """
        
        # Fallback ai parametri di classe se non passati
        eps = epsilon if epsilon is not None else getattr(self, 'epsilon', 0.05)
        dlt = delta if delta is not None else getattr(self, 'delta', 0.05)
        
        n = len(labels)
        if n == 0:
            return False
            
        vals, counts = np.unique(labels, return_counts=True)
        
        # Se il nodo ha una sola classe pura, si ferma subito
        if len(vals) == 1:
            return True
            
        # p_c: proporzione della classe maggioritaria
        pc = np.max(counts) / float(n)
        
        # Errore standard per l'approssimazione normale della binomiale
        std_err = np.sqrt((pc * (1.0 - pc)) / n)
        
        if std_err == 0:
            return True
            
        # Calcolo dello Z-score per la soglia (1 - epsilon)
        z_score = ((1.0 - eps) - pc) / std_err
        
        # CDF (Cumulative Distribution Function) calcola esattamente prob(p_c < (1 - epsilon))
        prob_less_than_threshold = scipy.stats.norm.cdf(z_score)
        
        # L'albero si ferma (True) solo se questa probabilità è inferiore al limite delta
        return bool(prob_less_than_threshold < dlt)

    def _check_leaf_dominance(self, y_labels):
        """
        Implementa il test di dominanza di TREPAN.
        Verifica se la classe maggioritaria domina la seconda classe
        con prob(p_c < p_j) < delta.
        """
        import numpy as np
        import scipy.stats
        
        n = len(y_labels)
        if n == 0:
            return False
            
        vals, counts = np.unique(y_labels, return_counts=True)
        if len(vals) == 1:
            return True  # Solo una classe, domina matematicamente
            
        # Ordina i conteggi in ordine decrescente
        sorted_counts = np.sort(counts)[::-1]
        
        pc = sorted_counts[0] / n  # Proporzione classe maggioritaria
        pj = sorted_counts[1] / n  # Proporzione seconda classe (rivale)
        
        # Varianza della differenza tra due proporzioni multinomiali
        var_diff = (pc + pj - (pc - pj)**2) / n
        
        if var_diff == 0:
            return True
            
        # Z-score: misura la distanza tra p_c e p_j
        z_score = (pc - pj) / np.sqrt(var_diff)
        
        # Calcolo di prob(p_c < p_j) valutando la coda sinistra (-z_score)
        prob_less = scipy.stats.norm.cdf(-z_score)
        
        return bool(prob_less < getattr(self, 'delta', 0.05))


    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
       
       current_node = getattr(self, '_current_node', None)
       return bool(getattr(current_node, 'is_statistically_pure', False))

    def queue_push(self, node: TrepanNode, idx: np.ndarray):
       
       if not hasattr(self, "queue"):

        self.queue = []

       if not hasattr(self, "tiebreaker"):
        


        self.tiebreaker = itertools.count()

       priority_value = -float(node.priority())
       heapq.heappush(self.queue, (priority_value, next(self.tiebreaker), idx, node))

    def make_split(self, X, y, X_ts=None, X_img=None, X_txt=None, idx=None, **kwargs):
        if hasattr(self, '_internal_nodes_count') and self._internal_nodes_count >= self.max_internal_nodes:  
              raise NoSplitFoundWarning("Limite nodi interni raggiunto")

        stump = super().make_split(X, y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, idx=idx, **kwargs)

        self._internal_nodes_count += 1
        return stump

        
    def _resolve_data_input(self, X=None, X_ts=None, X_img=None, X_txt=None):
        self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt)

        if X is not None:
            return X
        if X_ts is not None:
            return X_ts
        if X_img is not None:
            return X_img
        if X_txt is not None:
            return X_txt

        raise ValueError("At least one of X, X_ts, X_img or X_txt must be specified.")
    
    def _compute_importances(self, current_node=None, importances=None):
        return super()._compute_importances(current_node=current_node, importances=importances)
    
    def local_interpretation(self, X, joint_contribution=False):
        return super().local_interpretation(X, joint_contribution)
    
    def queue_pop(self):
        
        if not hasattr(self, "queue") or not self.queue:
            return None, None
            
        priority_value, tiebreaker, idx, current_node = heapq.heappop(self.queue)
        self._current_node = current_node
        
        return idx, current_node

    def _compute_sampling_params(self, idx, reach, constraints, m_of_n_rules):
        s_min = getattr(self, 's_min', 1000)
        total_samples = getattr(self, '_total_samples', 1000)
        n_real = len(idx)
        
        # 1. Calcolo proporzionale a s_min in base alla reach
        if reach < 0.01:
            n_max = max(50, int(s_min * 0.2))  
        elif reach < 0.05:
            n_max = max(100, int(s_min * 0.5)) 
        else:
            n_max = s_min                      
            
        n_init = max(30, int(n_max * 0.2))
        n_step = max(30, int(n_max * 0.2))
            
        # 2. Riduzione se abbiamo buon supporto di dati reali
        if n_real > 50:
            n_init = max(20, int(n_init * 0.5))
            n_step = max(20, int(n_step * 0.5))
            n_max = min(n_max, max(100, int(s_min * 0.5))) 
            
        # 3. Tetto massimo per dataset piccoli
        if total_samples < 200:
            n_max = min(n_max, max(int(total_samples), int(s_min * 0.5)))
            
        # 4. Penalizzazione per vincoli eccessivi
        if len(constraints) > 5 or len(m_of_n_rules) > 2:
            n_init = max(10, int(n_init * 0.5))
            n_step = max(10, int(n_step * 0.5))
            
        return n_init, n_step, n_max

    def print_trepan_rules(self, feature_names=None, digits=3, scaler=None):
        """
        Stampa le regole globali estratte da TREPAN.
        
        Args:
            feature_names (list): Nomi delle feature
            digits (int): Numero di decimali per i valori continui
            scaler (StandardScaler): Scaler usato per i dati di training (opzionale)
        """
        # FIX: Converte Index di pandas in lista Python nativa
        print("\n" + "="*70)
        print("  REGOLE GLOBALI ESTRATTE DA TREPAN")
        if scaler is not None:
            print("  (VALORI RISCALATI ALLA SCALA ORIGINALE)")
        print("="*70)
        
        if hasattr(self, 'opportunistic_node_map'):
            delattr(self, 'opportunistic_node_map')
            
        leaf_ids = self.get_leaf_nodes()
        if not leaf_ids:
            return

        # Funzione per riscalare un valore testuale al dataset
        def rescale_rule_text(rule_text, feature_names, scaler):
            import re
            if scaler is None or feature_names is None:
                return rule_text
            
            # Ordina le feature dalla più lunga alla più corta per evitare che 
            # match parziali rovinino i nomi (es. "age" che intercetta "parent_age")
            feature_names = list(feature_names)


            sorted_features = sorted(feature_names, key=len, reverse=True)
            
            for feat in sorted_features:
                idx = feature_names.index(feat)
                mean = scaler.mean_[idx]
                std = scaler.scale_[idx]
                
                # Se std è 0, la feature era costante, viene saltata
                if std == 0:
                    continue
                
                def repl(match):
                    op = match.group(1)                  # Cattura l'operatore (es. <=)
                    val_scaled = float(match.group(2))   # Cattura il valore scalato
                    
                    # Inversione della standardizzazione
                    val_orig = val_scaled * std + mean
                    
                    # Se il valore originario è vicinissimo a un intero, lo stampa pulito.
                    # Altrimenti, usa il parametro 'digits' per arrotondare.
                    if abs(val_orig - round(val_orig)) < 1e-4:
                        formatted_val = str(int(round(val_orig)))
                    else:
                        formatted_val = str(round(val_orig, digits))
                        
                    return f"{feat} {op} {formatted_val}"
                
                # Costruisce la regex dinamicamente. re.escape protegge caratteri come ( o )
                pattern = rf'{re.escape(feat)}\s*([<=>!]+)\s*([-+]?\d*\.?\d+)'
                rule_text = re.sub(pattern, repl, rule_text)
                
            return rule_text

        for i, leaf_id in enumerate(leaf_ids, 1):
            current_node = self.root
            path_conditions = []
            
            for char in str(leaf_id)[1:]:
                stump = getattr(current_node, 'stump', None)
                if stump is not None:
                    rule_dict = stump.get_rule(columns_names=feature_names, float_precision=digits)
                    if char == 'l':
                        rule_text = rule_dict.get('textual_rule', '')
                        if scaler is not None:
                            rule_text = rescale_rule_text(rule_text, feature_names, scaler)
                        path_conditions.append("(" + rule_text + ")")
                        current_node = getattr(current_node, 'node_l', None)
                    elif char == 'r':
                        rule_text = rule_dict.get('not_textual_rule', '')
                        if scaler is not None:
                            rule_text = rescale_rule_text(rule_text, feature_names, scaler)
                        path_conditions.append("(" + rule_text + ")")
                        current_node = getattr(current_node, 'node_r', None)
                else:
                    if char == 'l': current_node = getattr(current_node, 'node_l', None)
                    elif char == 'r': current_node = getattr(current_node, 'node_r', None)
                    
                if current_node is None:
                    break

            if current_node is None:
                continue

            if_clause = " AND \n      ".join(path_conditions) if path_conditions else "True (Radice pura)"
            pred = getattr(current_node, 'prediction', 'Sconosciuta')
            fid = getattr(current_node, 'fidelity', 0.0)
            rch = getattr(current_node, 'reach', 0.0)
            
            print(f"REGOLA {i} [Nodo ID: {leaf_id}]:")
            print(f"  IF  {if_clause}")
            print(f"  THEN Predizione = {pred}")
            print(f"  [Fedeltà: {fid:.2%}, Copertura: {rch:.2%}]\n")
