import heapq
import numpy as np
from typing import Any, Optional, Union

from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.tree.TrepanNode import TrepanNode, Constraint
from RuleTree.stumps.classification.TrepanStumpClassifier import TrepanStumpClassifier


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
                 s_min: int =1000):
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
        self.oracle = Oracle(estimator)
        self.estimator = estimator  # Keep a reference to the original estimator
        self.s_min = s_min

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
        
        # 1) Nodo base dal framework (calcola prediction, impurità, ecc.)
        base_node = super().prepare_node(y, idx, node_id, node)

        # 2) Calcolo di Fidelity / Reach
        if idx is None or len(idx) == 0:
            fidelity = 1.0
            reach = 0.0
        else:
            if getattr(self, '_X_train_temp', None) is None:
                reach = len(idx) / float(self._total_samples) if getattr(self, '_total_samples', 0) > 0 else 0.0
                fidelity = 0.0
            else:
                X_local = np.asarray(self._X_train_temp)[idx]
                y_pred_oracle = np.asarray(self.oracle.predict(X_local)).ravel()

                base_pred = base_node.prediction
                if np.ndim(base_pred) > 0:
                    base_arr = np.asarray(base_pred)
                    if base_arr.ndim == 1 and base_arr.size == len(base_arr):
                        if np.all(base_arr >= 0) and np.isclose(base_arr.sum(), 1.0):
                            base_label = int(np.argmax(base_arr))
                        else:
                            base_label = base_arr.ravel()[0]
                    else:
                        base_label = base_arr.ravel()[0]
                else:
                    base_label = base_pred

                fidelity = float(np.mean(y_pred_oracle == base_label))
                reach = float(len(idx)) / float(self._total_samples) if getattr(self, '_total_samples', 0) > 0 else 0.0

        # 3) COSTRUZIONE CONSTRAINTS (La Memoria Spaziale)
        node_constraints: list[Constraint] = []

        # Controlliamo se il nodo ha un padre (R è radice, Rl è figlio sinistro, ecc.)
        if node_id is not None and len(str(node_id)) > 1:
            try:
                # FIX ANTI-MEMORIA CORROTTA: Navighiamo l'albero partendo dalla radice 
                # anziché usare la mappa statica self._get_node()
                parent_node = self._get_parent_dynamically(node_id)
            except Exception:
                parent_node = None

            if parent_node is not None and isinstance(parent_node, TrepanNode):
                # Copia difensiva dei vincoli del padre
                node_constraints = parent_node.copy_constraints()

                # Se il padre ha uno stump, proviamo ad estrarre feature/threshold in modo difensivo
                padre_stump = getattr(parent_node, "stump", None)
                if padre_stump is not None:
                    try:
                        # Attributi usati abitualmente
                        feat = getattr(padre_stump, "feature_original", None)
                        thr = getattr(padre_stump, "threshold_original", None)

                        if feat is not None and thr is not None:
                            feat_idx = int(feat[0])
                            thr_val = float(thr[0])
                            
                            # Comprendiamo se siamo il figlio sinistro dalla stringa node_id (es. termina con 'l')
                            is_left = str(node_id).endswith('l')
                            
                            if is_left:
                                node_constraints.append(Constraint(feature_index=feat_idx, operator="<=", value=thr_val))
                            else:
                                node_constraints.append(Constraint(feature_index=feat_idx, operator=">", value=thr_val))
                        else:
                            # Fallback difensivo: proviamo altri nomi di attributo
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
                        # Se non riusciamo ad estrarre per problemi interni dello stump, andiamo avanti
                        pass

        # 4) Ritorna il TrepanNode incapsulando metriche e constraints spaziali
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
            constraints=node_constraints
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
        L'ANTIDOTO AL LOOP INFINITO.
        Ferma la crescita dell'albero se il nodo è puro secondo i dati reali 
        OPPURE se è puro secondo le previsioni dell'Oracolo (Fidelity 100%).
        """
        # 1. Condizione base nativa (dati reali puri)
        if len(np.unique(y[curr_idx])) <= 1:
            return True
        
        # 2. Condizione Trepan (Oracolo puro)
        if getattr(self, '_X_train_temp', None) is not None and len(curr_idx) > 0:
            try:
                X_local = self._X_train_temp[curr_idx]
                y_oracle = np.asarray(self.oracle.predict(X_local)).ravel()
                if len(np.unique(y_oracle)) <= 1:
                    return True  # L'oracolo è puro, fermati!
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
        print("  REGOLE GLOBALI ESTRATTE DA TREPAN")
        print("="*70)
        
        # --- FIX VITALE: Distruggiamo la cache corrotta del framework base ---
        if hasattr(self, 'opportunistic_node_map'):
            delattr(self, 'opportunistic_node_map')
            
        leaf_ids = self.get_leaf_nodes()
        if not leaf_ids:
            return

        for i, leaf_id in enumerate(leaf_ids, 1):
            leaf = self._get_node(leaf_id)
            conditions = []
            
            if hasattr(leaf, 'constraints') and leaf.constraints:
                for c in leaf.constraints:
                    feat_label = feature_names[c.feature_index] if feature_names is not None else f"Feature_{c.feature_index}"
                    val_str = f"{c.value:.{digits}f}" if isinstance(c.value, (int, float)) else str(c.value)
                    conditions.append(f"({feat_label} {c.operator} {val_str})")
            
            if_clause = " AND ".join(conditions) if conditions else "True (Radice pura)"
            pred = getattr(leaf, 'prediction', 'Sconosciuta')
            fid = getattr(leaf, 'fidelity', 0.0)
            rch = getattr(leaf, 'reach', 0.0)
            
            print(f"REGOLA {i} [Nodo ID: {leaf.node_id}]:")
            print(f"  IF  {if_clause}")
            print(f"  THEN Predizione = {pred}")
            print(f"  [Fedeltà: {fid:.2%}, Copertura: {rch:.2%}]\n") questo è il Trepanclassifier. per mofn va bene cosi come è? ce qualcosa da modificare per il nuovo stump?