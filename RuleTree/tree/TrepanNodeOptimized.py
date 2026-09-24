from typing import Optional
import numpy as np
from RuleTree.tree.RuleTreeNode import RuleTreeNode


class Constraint:
    """
    Rappresenta un vincolo geometrico su una feature.
    Esempio: Constraint(feature_index=5, operator="<=", value=0.5)
    """
    def __init__(self, feature_index, operator, value):
        self.feature_index = int(feature_index)
        self.operator = operator
        self.value = value

    def __repr__(self):
        return f"Constraint(feature={self.feature_index}, op={self.operator!r}, value={self.value!r})"


class TrepanNodeOptimized(RuleTreeNode):
    """
    Versione ottimizzata di TrepanNode.

    Ottimizzazione dello spazio:
    - NON memorizza le liste complete di constraints e m_of_n_rules.
    - Memorizza solo il proprio split (`split_info`) e un riferimento al padre.
    - Le liste complete vengono ricostruite on-demand risalendo l'albero.

    """

    def __init__(
        self,
        node_id: str,
        prediction,
        prediction_probability,
        log_odds,
        classes,
        parent: Optional[RuleTreeNode] = None,
        reach: float = 1.0,
        fidelity: float = 0.0,
        split_info: Optional[tuple] = None,   # ('constraint', Constraint) oppure ('m_of_n', (m, conditions, is_left))
        is_statistically_pure: bool = False,
        **kwargs
    ):
        super().__init__(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=prediction_probability,
            log_odds=log_odds,
            classes=classes,
            parent=parent,
            **kwargs
        )

        if not (0.0 <= fidelity <= 1.0):
            raise ValueError(f"fidelity must be in [0, 1], got {fidelity}")
        if reach < 0:
            raise ValueError(f"reach must be >= 0, got {reach}")

        self.fidelity = float(fidelity)
        self.reach = float(reach)
        self.split_info = split_info
        self.is_statistically_pure = bool(is_statistically_pure)


    def priority(self) -> float:
        """Priorità = Reach × (1 − Fidelity)."""
        return self.reach * (1.0 - self.fidelity)


    def get_split_data(self):
        """
        Risale l'albero dal nodo corrente alla radice e ricostruisce:
          - la lista dei Constraint geometrici (in ordine radice → nodo)
          - la lista delle regole M-of-N (in ordine radice → nodo)

        Una sola traversata restituisce entrambe le liste.
        """
        constraints = []
        rules = []
        node = self

        while node is not None:
            si = getattr(node, 'split_info', None)
            if si is not None:
                kind, data = si
                if kind == 'constraint':
                    constraints.append(data)
                else:  # kind == 'm_of_n'
                    rules.append(data)
            
            node = getattr(node, 'parent', None)

        # Le liste sono state costruite dal nodo verso la radice:
        # invertiamo per avere l'ordine radice → nodo.
        constraints.reverse()
        rules.reverse()
        return constraints, rules

    def get_constraints(self):
        """Restituisce solo la lista dei Constraint geometrici."""
        return self.get_split_data()[0]

    def get_m_of_n_rules(self):
        """Restituisce solo la lista delle regole M-of-N."""
        return self.get_split_data()[1]

    def copy_constraints(self):
        """
        Mantenuto per compatibilità con il codice esistente.
        Restituisce la lista dei Constraint ricostruita (non una copia di un attributo,
        perché l'attributo non esiste più).
        """
        return self.get_constraints()

    # ------------------------------------------------------------------
    # Confronto per la coda prioritaria
    # ------------------------------------------------------------------
    def __lt__(self, other):
        if not isinstance(other, TrepanNodeOptimized):
            return NotImplemented
        
        p_self = self.priority()
        p_other = other.priority()
        
        if p_self == p_other:
            return str(self.node_id) < str(other.node_id)
        
        return p_self < p_other

    def __repr__(self):
        return (
            f"TrepanNodeOptimized(id={self.node_id!r}, "
            f"pred={self.prediction!r}, "
            f"priority={self.priority():.6f}, "
            f"reach={self.reach:.6f}, "
            f"fidelity={self.fidelity:.6f}, "
            f"split_info={self.split_info!r})"
        )