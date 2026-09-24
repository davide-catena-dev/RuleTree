import numpy as np


def deep_sklearn_sizeof(obj, seen=None, with_overhead=True):
    """
    Stima la memoria profonda di un oggetto Python.

    Args:
        obj: oggetto da misurare.
        seen: set di id() già visitati (uso interno per la ricorsione).
        with_overhead: se True include l'header Python di ogni oggetto;
                       se False conta solo i dati puri.

    Returns:
        Dimensione stimata in byte.
    """
    if seen is None:
        seen = set()
    if obj is None:
        return 0

    # Anti-duplicazione
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Primitivi
    if isinstance(obj, (int, float, complex, bool, str, bytes)):
        return obj.__sizeof__()

    # NumPy scalars (np.float64, np.int64, ecc.)
    if isinstance(obj, np.generic):
        return obj.nbytes + (obj.__sizeof__() if with_overhead else 0)

    # Array NumPy
    if isinstance(obj, np.ndarray):
        return obj.__sizeof__() if with_overhead else obj.size * obj.itemsize

    # Contenitori standard 
    if isinstance(obj, (list, tuple, set, frozenset)):
        total = obj.__sizeof__() if with_overhead else 0
        for v in obj:
            total += deep_sklearn_sizeof(v, seen, with_overhead)
        return total

    if isinstance(obj, dict):
        total = obj.__sizeof__() if with_overhead else 0
        for k, v in obj.items():
            total += deep_sklearn_sizeof(k, seen, with_overhead)
            total += deep_sklearn_sizeof(v, seen, with_overhead)
        return total

    #  Oggetti custom (nodi, stump) 
    total = obj.__sizeof__() if with_overhead else 0

    # __slots__ (difensivo: nessuna classe RuleTree lo usa oggi)
    if hasattr(obj, '__slots__'):
        slots = obj.__slots__
        if isinstance(slots, str):
            slots = [slots]
        for slot in slots:
            try:
                v = getattr(obj, slot)
                total += deep_sklearn_sizeof(v, seen, with_overhead)
            except AttributeError:
                continue

    # __dict__ (caso principale)
    if hasattr(obj, '__dict__'):
        total += deep_sklearn_sizeof(obj.__dict__, seen, with_overhead)

    return total


# Conteggio struttura dell'albero
def count_tree_structure(root):
    """
    Conta la struttura di un albero RuleTree/TREPAN.

    Returns:
        dict con:
            - n_nodes: numero totale di nodi.
            - n_stumps: numero di nodi interni (con stump).
            - n_leaves: numero di foglie.
            - n_conditions: numero di condizioni M-of-N negli stump.
            - n_constraints: numero di constraint geometrici memorizzati.
            - n_rules_copies: numero di copie di regole M-of-N memorizzate.

    """
    counts = {
        'n_nodes': 0,
        'n_stumps': 0,
        'n_leaves': 0,
        'n_conditions': 0,
        'n_constraints': 0,
        'n_rules_copies': 0,
    }

    def recurse(node):
        if node is None:
            return
        counts['n_nodes'] += 1

        # --- Stump ---
        stump = getattr(node, 'stump', None)
        if stump is not None:
            counts['n_stumps'] += 1
            counts['n_conditions'] += len(getattr(stump, 'conditions', []) or [])
        else:
            counts['n_leaves'] += 1

        # --- Regole e constraints: SOLO UNO dei due (mutuamente esclusivi) ---
        si = getattr(node, 'split_info', None)
        if si is not None:
            # Nodo OTTIMIZZATO
            kind, data = si
            if kind == 'constraint':
                counts['n_constraints'] += 1
            elif kind == 'm_of_n':
                counts['n_rules_copies'] += 1
        else:
            # Nodo ORIGINALE
            counts['n_constraints'] += len(getattr(node, 'constraints', []) or [])
            counts['n_rules_copies'] += len(getattr(node, 'm_of_n_rules', []) or [])

        recurse(getattr(node, 'node_l', None))
        recurse(getattr(node, 'node_r', None))

    recurse(root)
    return counts