import heapq
import numpy as np
from typing import Any, Optional, Union
import torch 
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.tree.TrepanNode import TrepanNode, Constraint
from RuleTree.stumps.classification.TrepanStumpClassifier import TrepanStumpClassifier
import scipy.stats
from RuleTree.utils.synthetic_data import SyntheticDataGenerator

import numpy as np
from typing import Any, Optional

import numpy as np
from typing import Any, Optional

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
        # -------------------------------------------------------------
        # Adesso, se l'estimator era un modello PyTorch, qui dentro
        # self.estimator sarà l'InternalPyTorchWrapper, non il modello puro!
        # Quindi entrerà in questo primo 'if', trovando il metodo 'predict'
        # -------------------------------------------------------------
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
        self.model.eval() # Blocca gradienti e layer stocastici
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
                 delta : float = 0.05):
        # if no base_stumps provided, default to TrepanStumpClassifier to match _get_stumps_base_class
        if base_stumps is None:
            base_stumps = [TrepanStumpClassifier()]
        # Se per caso passi uno stump singolo, lo infiliamo in una lista per far felice il framework
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
        
        # -------------------------------------------------------------
        # INIEZIONE DEL WRAPPER INTERNO PER PYTORCH
        # Se l'estimator ha l'attributo 'parameters' tipico di nn.Module, 
        # lo avvolgiamo automaticamente per farlo dialogare con Scikit-Learn
        # -------------------------------------------------------------
        if hasattr(estimator, 'parameters'):
            estimator = InternalPyTorchWrapper(estimator)
            
        self.oracle = Oracle(estimator)
        self.estimator = estimator  # Questo ora punterà al wrapper se era un modello PyTorch
        self.s_min = s_min
        self.epsilon = epsilon
        self.delta = delta

    def _get_stumps_base_class(self):
        return TrepanStumpClassifier

    def fit(self, X: np.ndarray = None, y: np.ndarray = None,
            X_ts=None, X_img=None, X_txt=None, sample_weight=None, **kwargs):

        # 1. BLINDARE I DATI: Convertiamo tutto in array NumPy per distruggere gli indici di Pandas
        if X is not None:
            X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)

        # Keep a temporary reference while the base fit runs
        self._X_train_temp = X
        self._total_samples = float(self._X_train_temp.shape[0]) if self._X_train_temp is not None else 1.0
        
        # INIZIALIZZAZIONE CENTRALIZZATA DEL GENERATORE
        if self._X_train_temp is not None:
            self._synth_gen = SyntheticDataGenerator(self._X_train_temp, random_state=self.random_state)

        # Call base fit (ora X e y sono array NumPy puri e non daranno errori di KeyError)
        super().fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight, **kwargs)

        # Cleanup temporary storage
        self._X_train_temp = None
        return self
    
    
    def prepare_node(self,
                     y: np.ndarray,
                     idx: np.ndarray,
                     node_id: str,
                     node: Optional[TrepanNode] = None) -> TrepanNode:
        
        # 1) PREVENZIONE CRASH DA DATI SINTETICI:
        # Se lo split sintetico ha svuotato il ramo dei dati reali, ereditiamo dal padre
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
            
            # Esecuzione standard del framework se ci sono dati reali
            # CORREZIONE FEDELTÀ AL PAPER: L'oracolo etichetta i dati per la foglia
        # Assicuriamoci che _X_train_temp sia disponibile
            if self._X_train_temp is None: 
               raise RuntimeError("_X_train_temp non disponibile; chiamare fit() prima di prepare_node?")
        
            X_local = self._X_train_temp[idx]  # già array NumPy
            try:
                
               
                y_oracle_local = np.asarray(self.oracle.predict(X_local)).ravel()
                y_for_base = y.copy()
                y_for_base[idx] = y_oracle_local
            except Exception as e:
                
                
                # Fallback: usiamo le etichette originali, ma logghiamo l'errore
                import warnings
                warnings.warn(f"Oracle predict fallito in prepare_node: {e}. Uso etichette originali.")
                y_for_base = y
            base_node = super().prepare_node(y_for_base, idx, node_id, node)

        # 2) COSTRUZIONE CONSTRAINTS (La Memoria Spaziale)
        node_constraints: list[Constraint] = []
        node_m_of_n_rules = []

        if node_id is not None and len(str(node_id)) > 1:
            try:
                parent_node = self._get_parent_dynamically(node_id)
            except Exception:
                parent_node = None

            if parent_node is not None and isinstance(parent_node, TrepanNode):
                node_constraints = parent_node.copy_constraints()
                node_m_of_n_rules = list(getattr(parent_node, 'm_of_n_rules', []))

                padre_stump = getattr(parent_node, "stump", None)
                if padre_stump is not None:
                    if hasattr(padre_stump, 'conditions') and getattr(padre_stump, 'max_conditions', 1) > 1:
                        is_left = str(node_id).endswith('l')
                        node_m_of_n_rules.append((padre_stump.m, padre_stump.conditions, is_left))
                    else:
                        try:
                            feat = getattr(padre_stump, "feature_original", None)
                            thr = getattr(padre_stump, "threshold_original", None)
                            if feat is not None and thr is not None:
                                feat_idx = int(feat[0])
                                thr_val = float(thr[0])
                                is_left = str(node_id).endswith('l')
                                if is_left:
                                    node_constraints.append(Constraint(feature_index=feat_idx, operator="<=", value=thr_val))
                                else:
                                    node_constraints.append(Constraint(feature_index=feat_idx, operator=">", value=thr_val))
                            else:
                                feat_attr = getattr(padre_stump, "feature", None) or getattr(padre_stump, "feature_idx", None)
                                thr_attr = getattr(padre_stump, "threshold", None) or getattr(padre_stump, "thr", None)
                                if feat_attr is not None and thr_attr is not None:
                                    feat_idx = int(feat_attr[0]) if isinstance(feat_attr, (list, tuple, np.ndarray)) else int(feat_attr)
                                    thr_val = float(thr_attr[0]) if isinstance(thr_attr, (list, tuple, np.ndarray)) else float(thr_attr)
                                    is_left = str(node_id).endswith('l')
                                    if is_left:
                                        node_constraints.append(Constraint(feature_index=feat_idx, operator="<=", value=thr_val))
                                    else:
                                        node_constraints.append(Constraint(feature_index=feat_idx, operator=">", value=thr_val))
                        except Exception:
                            pass

        # 3) CALCOLO REACH E FIDELITY PURAMENTE MONTE CARLO (Paper TREPAN 1995)
        base_pred = base_node.prediction
        if np.ndim(base_pred) > 0:
            base_arr = np.asarray(base_pred)
            if base_arr.ndim == 1 and base_arr.size == len(base_arr):
                base_label = int(np.argmax(base_arr)) if (np.all(base_arr >= 0) and np.isclose(base_arr.sum(), 1.0)) else base_arr.ravel()[0]
            else:
                base_label = base_arr.ravel()[0]
        else:
            base_label = base_pred

        if getattr(self, '_synth_gen', None) is None:
            reach = 0.0
            fidelity = 0.0
        else:
            n_global_samples = getattr(self, 's_min', 1000)
            
            # --- FASE 3A: STIMA DEL REACH ---
            reach = self._synth_gen.estimate_reach(node_constraints, node_m_of_n_rules, n_global_samples)
            
            # --- FASE 3B: STIMA DELLA FIDELITY (Campionamento Adattivo Sequenziale) ---
            n_iniziale = 100
            n_step = 100
            n_max = 1000
            
            X_constrained = self._synth_gen.sample_constrained(n_iniziale, node_constraints, node_m_of_n_rules)
            
            if len(X_constrained) == 0:
                fidelity = 1.0 # Limite estremo: spazio collassato
            else:
                y_oracle_constrained = np.asarray(self.oracle.predict(X_constrained)).ravel()
                matches = np.sum(y_oracle_constrained == base_label)
                
                while len(X_constrained) < n_max:
                    n = len(X_constrained)
                    p = matches / n
                    
                    # Calcolo ampiezza intervallo di Wilson (al 95% di confidenza)
                    z = scipy.stats.norm.ppf(1 - self.delta) # Valore critico per alpha = 0.05
                    denominator = 1 + z**2 / n
                    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
                    
                    # Condizione di stop: Margine di errore < 2.5% (larghezza totale < 5%) o purezza estrema
                    if margin < 0.025 or p == 1.0 or p == 0.0:
                        break
                        
                    # Altrimenti, campioniamo un nuovo blocco per ridurre l'incertezza
                    X_extra = self._synth_gen.sample_constrained(n_step, node_constraints, node_m_of_n_rules)
                    if len(X_extra) == 0:
                        break # Impossibile generare altri dati, ci accontentiamo
                        
                    y_extra = np.asarray(self.oracle.predict(X_extra)).ravel()
                    matches += np.sum(y_extra == base_label)
                    X_constrained = np.vstack((X_constrained, X_extra))
                    
                fidelity = float(matches / len(X_constrained))

        # 4) Ritorna il TrepanNode
        return TrepanNode(
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
            m_of_n_rules=node_m_of_n_rules
        )
    
        
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
            
        # Saltiamo la 'R' iniziale e navighiamo 'l' o 'r' fino al padre
        for char in str(node_id)[1:-1]:
            if char == 'l':
                current = getattr(current, 'node_l', None)
            elif char == 'r':
                current = getattr(current, 'node_r', None)
            if current is None:
                return None
        return current
    
    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        """
        Criterio di arresto ATTIVO di TREPAN (1995).
        Se l'intervallo di confidenza sui dati reali non garantisce la purezza,
        interroga l'oracolo generando nuovi campioni per avere una conferma statistica.
        """
        if curr_idx is None or len(curr_idx) == 0:
            return True
            
        epsilon = getattr(self, 'epsilon', 0.05)
        delta = getattr(self, 'delta', 0.05)
        
        if getattr(self, '_X_train_temp', None) is None:
            return len(np.unique(y[curr_idx])) <= 1
            
        X_local = self._X_train_temp[curr_idx]
        try:
            y_oracle = np.asarray(self.oracle.predict(X_local)).ravel()
        except Exception:
            y_oracle = y[curr_idx]
            
        import scipy.stats
        
        def is_statistically_pure(labels):
            n = len(labels)
            if n == 0: return False
            vals, counts = np.unique(labels, return_counts=True)
            pc = np.max(counts) / n
            
            # Wilson Score Interval: Molto più robusto dell'approssimazione normale standard
            # Calcola correttamente il limite inferiore anche se pc = 1.0 (purezza 100%)
            z = scipy.stats.norm.ppf(1 - delta)
            denominator = 1 + z**2 / n
            centre_adjusted_prob = pc + z**2 / (2 * n)
            adjusted_standard_deviation = np.sqrt((pc * (1 - pc) + z**2 / (4 * n)) / n)
            
            lower_bound = (centre_adjusted_prob - z * adjusted_standard_deviation) / denominator
            
            # prob(pc < 1 - epsilon) < delta  => lower_bound >= 1 - epsilon
            return lower_bound >= (1.0 - epsilon)
            
        # 1. FASE PASSIVA: Verifichiamo se i dati reali sono già sufficienti
        if is_statistically_pure(y_oracle):
            return True
            
        # 2. FASE ATTIVA: Active Querying
        current_node = getattr(self, '_current_node', None)
        generator = getattr(self, '_synth_gen', None)
        
        if current_node is not None and generator is not None:
            try:
                constraints = getattr(current_node, 'constraints', [])
                m_of_n_rules = getattr(current_node, 'm_of_n_rules', [])
                n_needed = 500  
                
                # Generazione in una singola riga!
                X_synth = generator.sample_constrained(n_needed, constraints, m_of_n_rules)
                if len(X_synth) == 0:
                    # Nessun campione sintetico generabile: lo spazio è vuoto o degenere,
                    # quindi il nodo è considerato puro e ci fermiamo.
                    return True  
                    
                if len(X_synth) > 0:
                    y_synth = np.asarray(self.oracle.predict(X_synth)).ravel()
                    y_combined = np.concatenate((y_oracle, y_synth))
                    
                    if is_statistically_pure(y_combined):
                        return True
            except Exception:
                pass
                
        return False
    
    
    def queue_push(self, node: TrepanNode, idx: np.ndarray):
        """Inserisce il nodo nella coda prioritaria. Usa -priority per trasformare min-heap in max-heap."""
        if not hasattr(self, "queue"):
            self.queue = []
        if not hasattr(self, "tiebreaker"):
            import itertools
            self.tiebreaker = itertools.count()
        priority_value = -float(node.priority())
        heapq.heappush(self.queue, (priority_value, next(self.tiebreaker), idx, node))
        
    def _resolve_data_input(self, X=None, X_ts=None, X_img=None, X_txt=None):
        """Resolve the provided input container for RuleTree apply/local interpretation."""
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
        """Use the base RuleTree importance computation for TrepanClassifier."""
        return super()._compute_importances(current_node=current_node, importances=importances)

    
    def local_interpretation(self, X, joint_contribution=False):
        return super().local_interpretation(X, joint_contribution)
    
    def queue_pop(self):
        """Pop dalla coda e memorizza l'ultimo nodo estratto su self._current_node."""
        idx, current_node = super().queue_pop()
        # memorizziamo il nodo corrente per permettere agli stump di accedere ai constraints
        self._current_node = current_node
        return idx, current_node
    
    def print_trepan_rules(self, feature_names=None, digits=3):
        print("\n" + "="*70)
        print("  REGOLE GLOBALI ESTRATTE DA TREPAN")
        print("="*70)
        
        # --- FIX VITALE: Distruggiamo la cache corrotta del framework base ---
        if hasattr(self, 'opportunistic_node_map'):
            delattr(self, 'opportunistic_node_map')
            
        leaf_ids = self.get_leaf_nodes()
        if not leaf_ids:
            return

        for i, leaf_id in enumerate(leaf_ids, 1):
            # Navighiamo l'albero partendo dalla radice per ricostruire il percorso logico
            current_node = self.root
            path_conditions = []
            
            # str(leaf_id) è formato da R seguito da l/r. Es: "Rlrr"
            for char in str(leaf_id)[1:]:
                stump = getattr(current_node, 'stump', None)
                if stump is not None:
                    # Chiediamo allo stump di formattare la sua regola!
                    rule_dict = stump.get_rule(columns_names=feature_names, float_precision=digits)
                    if char == 'l':
                        path_conditions.append("(" + rule_dict.get('textual_rule', '') + ")")
                        current_node = getattr(current_node, 'node_l', None)
                    elif char == 'r':
                        path_conditions.append("(" + rule_dict.get('not_textual_rule', '') + ")")
                        current_node = getattr(current_node, 'node_r', None)
                else:
                    # Fallback di sicurezza in caso di rami monchi
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