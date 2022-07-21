from typing import List, Tuple, Dict, Set
import spot
from src.model import DFA
from pyeda.inter import *
from pyeda.boolalg.expr import * 
from pyeda.boolalg.expr import _One, _Zero

__author__ = "Stefan Klikovits <stefan@klikovits.net>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__    = "26 September 2020"

def is_safety(formula: str) -> bool:
    """Returns a bool to indicate whether a LTL formula is a safety formula."""
    return spot.formula(formula).is_syntactic_safety()


def spotGuardToDict(guard: str) -> List[Dict[str, bool]]:
    """
    This function takes a spot edge label (e.g. '!a & (b | !c)' )
    converts it into disjunctive normal form and splits it. (i.e. ['!a & b', '!a & !c'] )
    The individual splits are translated to property dicts: [ {'a': False, 'b': True}, {'a': False, 'c': False}]
    """
    parsed = expr(_spot2pyeda(guard))  # expr() is PyEDA's magic string to AST function...
    return _expr2dict(parsed.to_dnf())


def _spot2pyeda(formula: str) -> str:
    return formula.replace("!", "~")  # spot produces ! as negation, pyeda uses ~


def _expr2dict(node: Expression) -> List[Dict[str, bool]]:
    
    if isinstance(node, Variable):  # an AP
        assert node.depth == 0
        return [{node.name: True}]

    if isinstance(node, Complement):  # negated AP
        assert isinstance(~node, Variable)
        return [{(~node).name: False}]

    if isinstance(node, AndOp):
        result = {}
        for lit in node._lits:
            result.update(_expr2dict(lit)[0])
        return [result]

    if isinstance(node, OrOp):
        return [_expr2dict(lit)[0] for lit in node._lits]

    if isinstance(node, _One):  # _One is True
        return [{}]

    if isinstance(node, _Zero):  # _Zero is False
        return None
    breakpoint()

    raise Exception("Cannot deal with node " + str(node))


# Info: check here https://spot.lrde.epita.fr/tut21.html#orgc69204a
def ltl_to_dfa_spot(formula: str) -> DFA:
    aut = spot.translate(formula, 'monitor', 'det') 
    bdict = aut.get_dict()
    
    assert aut.get_acceptance() == spot.acc_code.t(), "Check that acceptance condition is True"  # true means that all runs that don't "die" are accepting
    assert aut.prop_universal() and aut.is_existential(), "Check that the DFA is deterministic"

    dfa = DFA([ap.to_str() for ap in aut.ap()])  # aut.ap() gives us the alphabet 
    dfa.setInitialState(aut.get_init_state_number() + 1)  #  we add +1 to each state number, since we don't start 

    for s in range(0, aut.num_states()):
        if aut.prop_state_acc().is_true() and aut.state_is_accepting(s):
            dfa.addSafeState(s + 1)
            
        for t in aut.out(s):
            label_string = spot.bdd_format_formula(bdict, t.cond)
            for g_dict in spotGuardToDict(label_string):
                dfa.addTransition(t.src + 1, g_dict, t.dst + 1)

    return dfa
