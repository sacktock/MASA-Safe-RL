class Atom: 
    """Atom: satisfied when the given atom is in the set of labels"""

    def __init__(self, atom):
        self.atom = atom

    def sat(self, labels):
        return self.atom in labels

class Truth: 
    """Truth: always satisfied"""

    def __init__(self):
        pass

    def sat(self, labels):
        return True

class And:
    """And: satisfied when both subformulae are satisfied"""

    def __init__(self, subformula_1, subformula_2):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels):
        return self.subformula_1.sat(labels) and self.subformula_2.sat(labels)

class Or:
    """Or: satisfied when either subformulae are satisfied"""

    def __init__(self, subformula_1, subformula_2):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels):
        return self.subformula_1.sat(labels) or self.subformula_2.sat(labels)

class Neg:
    """Negation: satisfied when the subformula is not satisfied"""
    
    def __init__(self, subformula):
        self.subformula = subformula
        
    def sat(self, labels):
        return not self.subformula.sat(labels)

class Implies:
    """Implies: satisfied when subformula_2 is satisified if subformula_1 is satisfied"""

    def __init__(self, subformula_1, subformula_2):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels):
        return Or(Neg(self.subformula_1), self.subformula_2).sat(labels)