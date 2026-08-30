import numpy as np
from scipy.stats import gaussian_kde
from sklearn.utils import check_random_state
from typing import Optional, List, Tuple, Callable

from RuleTree.tree.TrepanNode import Constraint

class SyntheticDataGenerator:
    """
    Generatore di dati sintetici per TREPAN.
    Modella le distribuzioni marginali delle feature a partire da X_train
    e genera campioni rispettando vincoli (constraints) e regole m-of-n.
    """
    def __init__(self, X_train: np.ndarray, random_state=None):
        self.X_train = np.asarray(X_train)
        self.rng = check_random_state(random_state)
        self.n_features = self.X_train.shape[1]
        
        # Pre-calcola le distribuzioni per ogni feature
        self._precompute_distributions()

    def _precompute_distributions(self):
        """Per ogni feature, salva valori unici, KDE, e se è categorica."""
        self.feature_info = []
        for f in range(self.n_features):
            col = self.X_train[:, f].astype(float)
            uniq, counts = np.unique(col, return_counts=True)
            is_cat = len(uniq) <= 10  # euristica
            
            kde = None
            if not is_cat and len(col) > 1:
                try:
                    # KDE con varianza aggiunta per evitare singolarità
                    kde = gaussian_kde(col)
                except np.linalg.LinAlgError:
                    kde = None
            
            self.feature_info.append({
                'is_cat': is_cat,
                'uniq': uniq,
                'counts': counts,
                'kde': kde,
                'min_val': float(np.min(col)),
                'max_val': float(np.max(col))
            })

    def _draw_feature_column(self, f_idx: int, n_samples: int, 
                             min_bound: Optional[float] = None,
                             max_bound: Optional[float] = None) -> np.ndarray:
        """
        Disegna n_samples per una singola feature, rispettando i bounds.
        """
        info = self.feature_info[f_idx]
        col_data = self.X_train[:, f_idx]
        
        # Fallback automatico se i bounds sono inconsistenti
        min_bound = max(min_bound, info['min_val']) if min_bound is not None else info['min_val']
        max_bound = min(max_bound, info['max_val']) if max_bound is not None else info['max_val']
        
        if min_bound > max_bound:
            # Se il range è vuoto, restituisci il valore centrale
            return np.full(n_samples, (min_bound + max_bound) / 2.0)

        if info['is_cat']:
            # Campionamento discreto pesato
            valid_mask = (info['uniq'] >= min_bound) & (info['uniq'] <= max_bound)
            valid_vals = info['uniq'][valid_mask]
            valid_counts = info['counts'][valid_mask]
            
            if len(valid_vals) == 0:
                # Se nessun valore valido, prendi il più vicino
                idx_closest = np.argmin(np.abs(info['uniq'] - (min_bound + max_bound)/2))
                return np.full(n_samples, info['uniq'][idx_closest])
            
            probs = valid_counts / valid_counts.sum()
            return self.rng.choice(valid_vals, size=n_samples, p=probs)
        
        else:
            # Feature continua: KDE o Uniform
            if info['kde'] is not None:
                try:
                    # Resample e clip ai bounds
                    samples = info['kde'].resample(n_samples, seed=self.rng)[0]
                    return np.clip(samples, min_bound, max_bound)
                except Exception:
                    pass
            
            # Fallback uniforme
            return self.rng.uniform(min_bound, max_bound, size=n_samples)

    def _apply_constraints(self, samples: np.ndarray, 
                           constraints: List[Constraint]) -> np.ndarray:
        """Filtra le righe che soddisfano tutti i vincoli."""
        if not constraints:
            return samples
        
        mask = np.ones(samples.shape[0], dtype=bool)
        for c in constraints:
            f_idx = c.feature_index
            val = c.value
            op = c.operator
            
            # Vettorizzazione rapida
            col = samples[:, f_idx]
            if op == "<=":
                mask &= (col <= val)
            elif op == ">=":
                mask &= (col >= val)
            elif op == "==":
                mask &= (col == val)
            elif op == "!=":
                mask &= (col != val)
            elif op == ">":
                mask &= (col > val)
            elif op == "<":
                mask &= (col < val)
        return samples[mask]

    def _apply_m_of_n_rules(self, samples: np.ndarray, 
                            m_of_n_rules: List[Tuple[int, List, bool]]) -> np.ndarray:
        """Filtra le righe che soddisfano le regole m-of-n del nodo."""
        if not m_of_n_rules:
            return samples
        
        mask = np.ones(samples.shape[0], dtype=bool)
        for m, conditions, is_left in m_of_n_rules:
            # Calcola il conteggio delle condizioni soddisfatte per ogni campione
            satisfied_counts = np.zeros(samples.shape[0], dtype=int)
            for feat_idx, thresh, op in conditions:
                col = samples[:, feat_idx]
                if op == "<=":
                    satisfied_counts += (col <= thresh).astype(int)
                elif op == "==":
                    satisfied_counts += (col == thresh).astype(int)
                elif op == "!=":
                    satisfied_counts += (col != thresh).astype(int)
                elif op == ">":
                    satisfied_counts += (col > thresh).astype(int)
            
            # La regola richiede che il campione soddisfi almeno m condizioni
            # OPPURE che non le soddisfi, a seconda di `is_left`
            rule_ok = (satisfied_counts >= m)
            if not is_left:
                rule_ok = ~rule_ok
            mask &= rule_ok
        
        return samples[mask]

    def generate(self, n_samples: int, 
                 constraints: Optional[List[Constraint]] = None,
                 m_of_n_rules: Optional[List[Tuple[int, List, bool]]] = None,
                 return_all: bool = False) -> np.ndarray:
        """
        Genera n_samples sintetici rispettando constraints e m-of-n-rules.
        
        Se dopo i filtri rimangono meno di n_samples, esegue un resampling 
        con reinserimento per raggiungere esattamente n_samples.
        Se return_all=True, restituisce TUTTI i campioni filtrati (senza riempimento).
        """
        constraints = constraints or []
        m_of_n_rules = m_of_n_rules or []
        
        # 1. Calcola i bounds globali partendo dai vincoli
        mins = np.array([info['min_val'] for info in self.feature_info])
        maxs = np.array([info['max_val'] for info in self.feature_info])
        
        for c in constraints:
            f_idx = c.feature_index
            val = float(c.value)
            if c.operator in ("<=", "<"):
                maxs[f_idx] = min(maxs[f_idx], val)
            elif c.operator in (">=", ">"):
                mins[f_idx] = max(mins[f_idx], val)
            elif c.operator == "==":
                mins[f_idx] = max(mins[f_idx], val)
                maxs[f_idx] = min(maxs[f_idx], val)
        
        # 2. Genera un pool iniziale (5x per avere abbastanza campioni dopo i filtri)
        pool_size = n_samples * 5
        samples = np.zeros((pool_size, self.n_features))
        for f in range(self.n_features):
            samples[:, f] = self._draw_feature_column(f, pool_size, mins[f], maxs[f])
        
        # 3. Applica filtri (prima vincoli, poi m-of-n)
        samples = self._apply_constraints(samples, constraints)
        samples = self._apply_m_of_n_rules(samples, m_of_n_rules)
        
        # 4. Gestione del caso vuoto (FIX DEL BUG!)
        if len(samples) == 0:
            if return_all:
                return np.empty((0, self.n_features))
            # Se non return_all, riproviamo con un campionamento meno restrittivo:
            # Generiamo uniformemente nella bounding box senza KDE per forzare la presenza
            samples = np.zeros((n_samples, self.n_features))
            for f in range(self.n_features):
                samples[:, f] = self.rng.uniform(mins[f], maxs[f], size=n_samples)
            # Riapplichiamo i filtri
            samples = self._apply_constraints(samples, constraints)
            samples = self._apply_m_of_n_rules(samples, m_of_n_rules)
            
            if len(samples) == 0 and not return_all:
                # Ultima spiaggia: restituisci campioni che soddisfano solo i constraints (ignora m-of-n)
                samples = np.zeros((n_samples, self.n_features))
                for f in range(self.n_features):
                    samples[:, f] = self.rng.uniform(mins[f], maxs[f], size=n_samples)
                samples = self._apply_constraints(samples, constraints)
                # Se ancora zero, prendi i centroidi
                if len(samples) == 0:
                    samples = np.tile((mins + maxs) / 2, (n_samples, 1))
        
        # 5. Tronca o riempie per avere esattamente n_samples
        if return_all:
            return samples
        
        if len(samples) >= n_samples:
            return samples[:n_samples]
        else:
            # Resampling con reinserimento
            indices = self.rng.choice(len(samples), size=n_samples, replace=True)
            return samples[indices]
        
    def estimate_reach(self, constraints, m_of_n_rules, n_samples=1000):
        """Stima la frazione di spazio che soddisfa i vincoli."""
        samples = self.generate(n_samples, constraints, m_of_n_rules, return_all=True)
        return len(samples) / float(n_samples)
    
    def sample_constrained(self, n_samples, constraints, m_of_n_rules):  
        """Alias per generate con return_all=False."""
        return self.generate(n_samples, constraints, m_of_n_rules, return_all=False)