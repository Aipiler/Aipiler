from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from Aipiler.primitive import (
    EinsumPrimitive,
    MapPrimitive,
    ReducePrimitive,
    PopulatePrimitive,
    UnaryPrimitive,
)
from mlir import ir


class Visitor(ABC):
    @abstractmethod
    def visit_map(self, node: MapPrimitive) -> Any:
        pass

    @abstractmethod
    def visit_reduce(self, node: ReducePrimitive) -> Any:
        pass

    @abstractmethod
    def visit_populate(self, node: PopulatePrimitive) -> Any:
        pass

    @abstractmethod
    def visit_unary(self, node: UnaryPrimitive) -> Any:
        pass


class MLIRCodeGenVisitor(Visitor):
    def __init__(self, context: ir.Context):
        self.visited_nodes: List[EinsumPrimitive] = []

    def visit_map(self, node: MapPrimitive) -> Any:
        self.visited_nodes.append(node)

        return node.output

    def visit_reduce(self, node: ReducePrimitive) -> Any:
        self.visited_nodes.append(node)
        return node.output

    def visit_populate(self, node: PopulatePrimitive) -> Any:
        self.visited_nodes.append(node)
        return node.output

    def visit_unary(self, node: UnaryPrimitive) -> Any:
        self.visited_nodes.append(node)
        return node.output
