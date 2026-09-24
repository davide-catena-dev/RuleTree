from typing import Optional, Iterable, Union
import numpy as np
from RuleTree.tree.RuleTreeNode import RuleTreeNode

class Constraint:
    """
    Rappresenta un singolo vincolo su una feature.
    Esempio:
        Constraint(5, "<=", 0.5)
        Constraint(2, "==", "red")
    """

    def __init__(self, feature_index: int, operator: str, value: Union[int, float, str, tuple, list]):
        self.feature_index = int(feature_index)
        self.operator = operator
        self.value = value


    def __repr__(self) -> str:
        return f"Constraint(feature={self.feature_index}, op={self.operator!r}, value={self.value!r})"
    

class TrepanNode(RuleTreeNode):
    """
    Nodo specializzato per Trepan.
    """

    def __init__(
        self,
        node_id: str,
        prediction: Union[int, str, float],
        prediction_probability: Union[np.ndarray, float],
        log_odds: Union[np.ndarray, float],
        classes: np.ndarray,
        parent: Optional["RuleTreeNode"] = None,
        reach: float = 1.0,
        fidelity: float = 0.0,
        constraints: Optional[Iterable[Constraint]] = None,
        m_of_n_rules: Optional[list] = None,
        is_statistically_pure: int=False,
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
            raise ValueError(f"fidelity must be in [0,1], got {fidelity}")
        if reach < 0:
            raise ValueError(f"reach must be >= 0, got {reach}")

        self.fidelity = float(fidelity)
        self.reach = float(reach)

        # Copia difensiva: così un figlio non modifica accidentalmente la lista del padre
        self.constraints = list(constraints) if constraints is not None else []
        self.m_of_n_rules = list(m_of_n_rules) if m_of_n_rules is not None else []
        self.is_statistically_pure = bool(is_statistically_pure)

    def priority(self) -> float:
        """Priorità TREPAN: più reach, meno fidelity => più priorità."""
        return self.reach * (1.0 - self.fidelity)


    def copy_constraints(self) -> list[Constraint]:
        """
        Restituisce una copia dei constraints del nodo.
        Utile quando si crea un figlio.
        """
        return list(self.constraints)

    

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TrepanNode):
            return NotImplemented
        p_self = self.priority()
        p_other = other.priority()
        if p_self == p_other:
            return str(self.node_id) < str(other.node_id)
        return p_self < p_other

    def __repr__(self) -> str:
        return (
            f"TrepanNode(id={self.node_id!r}, "
            f"pred={self.prediction!r}, "
            f"priority={self.priority():.6f}, "
            f"reach={self.reach:.6f}, "
            f"fidelity={self.fidelity:.6f}, "
            f"constraints={self.constraints!r}, "
            f"m_of_n_rules={self.m_of_n_rules!r})"
        )
        
        