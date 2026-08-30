import heapq
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any, Callable, List, Optional, Tuple, Union
from RuleTree.tree import RuleTree

class Oracle:
    """Wrapper per uniformare l'interrogazione di qualsiasi modello black-box.
    
    Garantisce che il modello restituisca sempre array NumPy coerenti,
    indipendentemente dal framework di origine (Scikit-Learn, PyTorch, TensorFlow).
    """
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Restituisce le classi predette dall'oracolo come array NumPy."""
        if hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X)
        elif callable(self.estimator):
            preds = self.estimator(X)
        else:
            raise TypeError("L'estimatore fornito non ha un metodo 'predict' né è invocabile direttamente.")
        
        # Gestione di tensori PyTorch o array di altri framework
        if hasattr(preds, "detach"):  # Se è un tensore PyTorch con gradiente
            preds = preds.detach().cpu().numpy()
        elif hasattr(preds, "numpy"):  # Se è un tensore o oggetto con metodo .numpy()
            preds = preds.numpy()
            
        return np.asarray(preds)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Restituisce le probabilità o distribuzioni di classe se il modello lo supporta."""
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            if hasattr(probs, "detach"):
                probs = probs.detach().cpu().numpy()
            elif hasattr(probs, "numpy"):
                probs = probs.numpy()
            return np.asarray(probs)
        return None


class TrepanNode:
    """Rappresenta un singolo nodo dell'albero di decisione generato da Trepan.
    
    Implementa l'operatore relazionale __lt__ invertito per permettere a heapq
    di comportarsi come un Max-Heap (estrarrà sempre il nodo a priorità più alta).
    """
    def __init__(self, depth: int = 0, parent: Optional['TrepanNode'] = None):
        self.depth: int = depth
        self.parent: Optional['TrepanNode'] = parent
        self.children: List['TrepanNode'] = []
        self.data: Optional[np.ndarray] = None
        # Stato e regole del nodo
        self.is_leaf: bool = False
        self.predicted_class: Optional[int] = None
        self.feature_index: Optional[int] = None  # Per split univariati semplici (iniziale)
        self.threshold: Optional[float] = None
        self.split_rule: Optional[Callable[[np.ndarray], np.ndarray]] = None  # Per future regole M-of-N
        
        # Metriche per l'espansione Best-First
        self.priority: float = 0.0
        self.reach: float = 1.0  # Percentuale di istanze che raggiungono questo nodo
        self.fidelity: float = 0.0  # Fedeltà locale rispetto all'Oracolo

    def __lt__(self, other: 'TrepanNode') -> bool:
        # FONDAMENTALE: l'inversione di '>' in < costringe heapq a fare un Max-Heap
        return self.priority > other.priority


class TrepanClassifier(RuleTree, ClassifierMixin):
    """Classificatore basato su Alberi Interpretabili estratto tramite algoritmo Trepan.
    
    Args:
        estimator: Il modello black-box addestrato da spiegare.
        max_tree_size: Numero massimo totale di nodi da espandere nell'albero.
        min_samples: Numero minimo di campioni (reali + sintetici) richiesti per uno split.
        max_depth: Profondità massima consentita per l'albero.
        random_state: Seed per la riproducibilità dei dati sintetici.
    """
    def __init__(
        self,
        estimator: Any = None,
        max_tree_size: int = 15,
        min_samples: int = 100,
        max_depth: int = 5,
        random_state: Optional[int] = None
    ):
        self.estimator = estimator
        self.max_tree_size = max_tree_size
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Variabili interne popolate durante il fit
        self.oracle_: Optional[Oracle] = None
        self.root_: Optional[TrepanNode] = None
        self.n_nodes_: int = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TrepanClassifier':
        """Costruisce l'albero Trepan interrogando l'estimatore fornito.
        
        Nota: 'y' è opzionale perché Trepan impara dalle predizioni dell'Oracolo,
        non dalle etichette reali del dataset.
        """
        if self.estimator is None:
            raise ValueError("Devi passare un modello 'estimator' addestrato per avviare Trepan.")
            
        X_train = np.asarray(X)
        self.oracle_ = Oracle(self.estimator)
        
        # 1. Inizializzazione della radice e della Coda di Priorità
        self.root_ = TrepanNode(depth=0)
        self.root_.reach = 1.0
        self.root_.data = X_train
        self._evaluate_node_priority(self.root_, X_train)
        
        priority_queue: List[TrepanNode] = []
        heapq.heappush(priority_queue, self.root_)
        self.n_nodes_ = 1

        # 2. Ciclo principale Best-First
        while priority_queue and self.n_nodes_ < self.max_tree_size:
            current_node = heapq.heappop(priority_queue)
            
            # Condizione di arresto per profondità o fedeltà già perfetta
            if current_node.depth >= self.max_depth or current_node.fidelity == 1.0 or (self.n_nodes_+ 2 > self.max_tree_size):
                self._make_leaf(current_node, current_node.data)
                continue

            # TODO Step 2: Generazione Dati Sintetici se len(X_node) < self.min_samples
            
            X_node = current_node.data
            y_oracle_node = self.oracle_.predict(X_node)
            best_col, best_thr, best_score = self._find_best_split(X_node, y_oracle_node)
            if best_col is  None or best_score <= 0:
                self._make_leaf(current_node,X_node)
                continue
            
            current_node.feature_index = best_col
            current_node.threshold = best_thr
            current_node.is_leaf = False
            current_node.predicted_class = None
            mask_left = X_node[:, best_col] <= best_thr
            X_left = X_node[mask_left]
            
            X_right = X_node[~mask_left] 
            if len(X_left) == 0 or len(X_right) == 0:
                self._make_leaf(current_node, X_node)
                continue
            
            w_left = len(X_left) / len(X_node)
            w_right = len(X_right) / len(X_node)
            reach_left = current_node.reach * w_left
            reach_right = current_node.reach * w_right
            left_child = TrepanNode(parent=current_node, depth=current_node.depth + 1)
            right_child = TrepanNode(parent=current_node, depth=current_node.depth + 1)
            left_child.reach = reach_left
            right_child.reach = reach_right
            left_child.data = X_left
            right_child.data = X_right
            self._evaluate_node_priority(left_child, X_left)
            self._evaluate_node_priority(right_child, X_right)
            current_node.children.append(left_child)
            current_node.children.append(right_child)
            current_node.data = None
            
            self.n_nodes_ += 2
            heapq.heappush(priority_queue, left_child)
            heapq.heappush(priority_queue, right_child)
            
        # 3. Chiusura finale: trasformiamo in foglie tutti i nodi rimasti nella coda!
        while priority_queue:  
            survivor_node = heapq.heappop(priority_queue)
            self._make_leaf(survivor_node, survivor_node.data)
            survivor_node.data = None  # Liberiamo anche qui la memoria!

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Attraversa l'albero per generare le predizioni."""
        if self.root_ is None:
            raise RuntimeError("Il modello non è ancora stato addestrato. Chiama .fit() prima di .predict().")
        
        X_test = np.asarray(X)
        predictions = []
        for instance in X_test:
            node = self.root_
            while not node.is_leaf:
                if not node.children or len(node.children) < 2:
                    break
                feature_idx = node.feature_index
                threshold = node.threshold
                if feature_idx is None or threshold is None:
                    break
                if instance[feature_idx] <= threshold:
                    
                    node = node.children[0]  # Il primo elemento è il left_child
                else:
                    
                    node = node.children[1]
                    
            pred = node.predicted_class if node.predicted_class is not None else 0
            predictions.append(pred)
        return np.array(predictions)

    def _evaluate_node_priority(self, node: TrepanNode, X_node: np.ndarray) -> None:
        """Calcola la priorità del nodo: Reach(N) * (1 - Fidelity(N))."""
        if len(X_node) == 0:
            node.priority = 0.0
            return
            
        oracle_preds = self.oracle_.predict(X_node)
        # Calcola la classe di maggioranza secondo l'oracolo
        vals, counts = np.unique(oracle_preds, return_counts=True)
        majority_class = vals[np.argmax(counts)]
        
        # La fedeltà locale è la precisione del nodo se fosse una foglia con la classe di maggioranza
        node.fidelity = np.mean(oracle_preds == majority_class)
        node.predicted_class = int(majority_class)
        
        # Priorità: quanto guadagno potenziale di fedeltà ci offre espandere questo nodo?
        node.priority = node.reach * (1.0 - node.fidelity)

    def _make_leaf(self, node: TrepanNode, X_node: np.ndarray) -> None:
        
        
        """Rende il nodo una foglia definitiva, garantendo che abbia sempre una classe predetta."""
        node.is_leaf = True
    
        # Se la classe è già stata assegnata in precedenza, non facciamo nulla
        if node.predicted_class is not None:
            return
        

       # CASO NORMALE: Il nodo ha dei dati suoi per determinare la maggioranza
        if X_node is not None and len(X_node) > 0:
            oracle_preds = self.oracle_.predict(X_node)
            vals, counts = np.unique(oracle_preds, return_counts=True)
            node.predicted_class = int(vals[np.argmax(counts)])
    
        # CASO DI EMERGENZA: Il nodo è rimasto senza dati! (Empty Splitting / Edge Case)
        else:
            
            # Se ha un nodo padre con una classe già decisa o da decidere, proviamo a ereditare da lui
            if node.parent is not None and node.parent.predicted_class is not None:
                   
                node.predicted_class = node.parent.predicted_class
            else:
                
                # Extrema ratio assoluta: assegniamo 0 di default così da non crashare mai con un NoneType!
                node.predicted_class = 0
            
    def _calculate_entropy(self, y_oracle: np.ndarray) -> float:
        _, counts = np.unique(y_oracle, return_counts=True)
        p = counts / len(y_oracle)
        p = p[p > 0]
        n = -(np.sum( p * np.log2(p)))
        return n
        
    def _calculate_gain_ratio(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        if (len(y_left) == 0) or (len(y_right) == 0):
            return 0.0
        total_samples = len(y_parent)
        w_left = len(y_left) / total_samples
        w_right = len(y_right) / total_samples
        # Entropia del padre - (media pesata delle entropie dei due figli)
        info_gain = self._calculate_entropy(y_parent) - (w_left * self._calculate_entropy(y_left) + w_right * self._calculate_entropy(y_right))
        if info_gain <= 0:
            return 0.0
        split_info = -(w_left * np.log2(w_left) + w_right * np.log2(w_right))
        if split_info < 1e-7:
            return 0.0
        else:
            return info_gain / split_info
        
    def _find_best_split(self, X_node: np.ndarray, y_oracle: np.ndarray):
        
        # BLOCCO 1: Inizializzazione dei campioni del torneo
        # Partiamo da un punteggio negativo così qualsiasi taglio reale (anche 0.0) potrà batterlo
        best_gain_ratio = -1.0
        best_col = None
        best_threshold = None
    
        # Ricava il numero di colonne dalla forma (shape) della matrice X_node
        _, n_features = X_node.shape
    
        # BLOCCO 2: Ciclo esterno - Esplorazione di tutte le feature
        for col_idx in range(n_features):
            # 1. Estrai la colonna corrente da X_node
            # 2. Usa np.unique() per trovare i valori unici ordinati presenti in questa colonna
            valori_unici = np.unique(X_node[:, col_idx])
    
        
        # Protezione: se tutti i campioni hanno lo stesso identico valore per questa feature,
        # è impossibile dividerli. Salta subito alla prossima colonna con 'continue'!
            if len(valori_unici) <= 1:
                
                continue
            
            # BLOCCO 3: Calcolo delle soglie candidate
            # Applica il trucco vettoriale per trovare i punti medi tra i valori unici adiacenti
            soglie = (valori_unici[:-1] + valori_unici[1:]) / 2
        
            # Ciclo interno - Test di ogni singola soglia
            for thr in soglie:
                
                
                # 1. Crea la maschera booleana (es. colonna <= thr)
                mask_left = X_node[:, col_idx] <= thr
                # 2. Usa la maschera per ricavare y_left e, con la tilde (~), ricava y_right
                y_left = y_oracle[mask_left]
                y_right = y_oracle[~mask_left]
                # 3. Chiama self._calculate_gain_ratio(y_oracle, y_left, y_right) e salva il punteggio
                score = self._calculate_gain_ratio(y_oracle,y_left,y_right)
            
                # BLOCCO 4: La resa dei conti
                # Se il punteggio appena calcolato batte il best_gain_ratio attuale:
                # - sovrascrivi best_gain_ratio con il nuovo score
                # - salva col_idx come best_col
                # - salva thr come best_threshold
                if score > best_gain_ratio:
                    best_gain_ratio = score
                    best_col = col_idx
                    best_threshold = thr
                
        # Alla fine di tutti i cicli, restituiamo i dati del vincitore assoluto
        return best_col, best_threshold, best_gain_ratio