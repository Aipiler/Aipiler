from .rankVariable import RankVariable
from typing import List, Optional, Union, Callable, Dict, Any, Tuple, Set

# --- Term classes ---


class ConstTerm:
    """Represents a constant term (e.g., 2)."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Get the value of the constant term."""
        return self.value

    def __repr__(self):
        return str(self.value)


class VarTerm:
    """Represents a variable term with a coefficient (e.g., 2s)."""

    def __init__(self, variable: RankVariable, coefficient: int = 1):
        self.variable = variable
        self.coefficient = coefficient

    def get_variable(self) -> RankVariable:
        """Get the variable associated with this term."""
        return self.variable

    def get_coefficient(self) -> int:
        """Get the coefficient of this term."""
        return self.coefficient

    def __repr__(self):
        if self.coefficient == 1:
            return f"{self.variable}"
        elif self.coefficient == -1:
            return f"-{self.variable}"
        else:
            return f"{self.coefficient}{self.variable}"


class AffineTerm:
    """Represents a term in an affine expression."""

    def __init__(self, constTerm: ConstTerm, varTerms: List[VarTerm]):
        """
        Initialize an affine term with a constant and variable terms.

        Args:
            constTerm: The constant term (default: 0)
            *varTerms: Variable number of VarTerm objects
        """
        self.varTerms = varTerms
        self.constTerm = constTerm

    def get_var_terms(self) -> List[VarTerm]:
        """Get all variable terms."""
        return self.varTerms

    def get_const_term(self) -> ConstTerm:
        """Get the constant term."""
        return self.constTerm

    def get_variables(self) -> List[RankVariable]:
        """Get all variables in the affine expression."""
        return [varTerm.get_variable() for varTerm in self.varTerms]

    def add_var_term(self, varTerm: VarTerm):
        """Add a variable term to the affine expression."""
        self.varTerms.append(varTerm)

    def __repr__(self):
        """String representation of the affine term."""
        result = ""

        # Add variable terms
        if self.varTerms:
            result = str(self.varTerms[0])
            for term in self.varTerms[1:]:
                if term.get_coefficient() >= 0:
                    result += f" + {term}"
                else:
                    # For negative coefficients, the minus sign is already included in the term's string representation
                    result += f" {term}"

        # Add constant term if non-zero
        constant = self.constTerm.get_value()
        if constant != 0:
            if constant > 0:
                prefix = " + " if result else ""
                result += f"{prefix}{constant}"
            else:
                result += f" - {abs(constant)}"

        # Return "0" if the expression is empty
        return result if result else "0"
