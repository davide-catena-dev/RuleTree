from typing import Optional, Iterable, Union
import numpy as np
from RuleTree.tree.RuleTreeNode import RuleTreeNode
import copy

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

    def satisfies(self, sample: np.ndarray) -> bool:
        """
        Controlla se un singolo campione soddisfa il vincolo.
        sample deve essere un array 1D, dove sample[i] è il valore della feature i.
        """
        if self.feature_index < 0 or self.feature_index >= len(sample):
            raise IndexError(f"feature_index {self.feature_index} out of bounds for sample of size {len(sample)}")

        x = sample[self.feature_index]

        if self.operator == "<=":
            return x <= self.value
        if self.operator == "<":
            return x < self.value
        if self.operator == ">=":
            return x >= self.value
        if self.operator == ">":
            return x > self.value
        if self.operator == "==":
            return x == self.value
        if self.operator == "!=":
            return x != self.value
        if self.operator == "in":
            return x in self.value

        raise ValueError(f"Unsupported operator: {self.operator}")

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

    def extend_constraints(self, new_constraints: Union[Constraint, Iterable[Constraint]]) -> None:
        """
        Aggiunge uno o più vincoli alla lista del nodo.
        """
        if isinstance(new_constraints, Constraint):
            self.constraints.append(new_constraints)
        else:
            self.constraints.extend(list(new_constraints))

    def copy_constraints(self) -> list[Constraint]:
        """
        Restituisce una copia dei constraints del nodo.
        Utile quando si crea un figlio.
        """
        return list(self.constraints)

    def match_constraints(self, sample: np.ndarray) -> bool:
        """
        Controlla se un singolo campione soddisfa tutti i vincoli del nodo.
        """
        for constraint in self.constraints:
            if not constraint.satisfies(sample):
                return False
        return True

    def match_constraints_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Controlla tutti i campioni in X e restituisce un array booleano.
        """
        return np.array([self.match_constraints(sample) for sample in X], dtype=bool)
    
    
   
    def satisfies_m_of_n_rules(self, sample: np.ndarray) -> bool:
        """
        Verifica se un campione rispetta tutte le regole M-of-N storiche 
        ereditate dai nodi antenati lungo il percorso.
        """
        for m, conditions, is_left in self.m_of_n_rules:
            satisfied_count = 0
            for feat_idx, thresh, op in conditions:
                val = sample[feat_idx]
                if op == "<=": satisfied_count += int(val <= thresh)
                elif op == "==": satisfied_count += int(val == thresh)
                elif op == "!=": satisfied_count += int(val != thresh)
                elif op == ">": satisfied_count += int(val > thresh)
            
            # Se is_left è True, il campione doveva soddisfare >= m condizioni
            node_requirement = (satisfied_count >= m)
            if node_requirement != is_left:
                return False
        return True
    

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
        
        
    def make_child(self, new_constraint: Constraint) -> "TrepanNode":

        """
        Restituisce un nuovo nodo figlio
        """
        new_constraints = self.copy_constraints()
        new_constraints.append(new_constraint)
        child_m_of_n_rules = copy.deepcopy(getattr(self, 'm_of_n_rules', []))
        
        return TrepanNode(
            node_id=f"{self.node_id}_child",
            prediction=self.prediction,
            prediction_probability=self.prediction_probability,
            log_odds=self.log_odds,
            classes=self.classes,
            parent=self,
            reach=self.reach,  # Il reach del figlio può essere calcolato successivamente
            fidelity=self.fidelity,  # La fidelity del figlio può essere calcolata successivamente
            constraints=new_constraints,
            m_of_n_rules=child_m_of_n_rules
        )
    