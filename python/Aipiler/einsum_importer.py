from iree.compiler.ir import (
    AffineAddExpr,
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineModExpr,
    AffineMulExpr,
    AffineSymbolExpr,
    Attribute,
    Block,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    FloatAttr,
    BF16Type,
    ComplexType,
    Float8E5M2Type,
    Float8E4M3FNType,
    Float8E5M2FNUZType,
    Float8E4M3FNUZType,
    F16Type,
    F32Type,
    F64Type,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    RankedTensorType,
    Location,
    Module,
    Operation,
    StringAttr,
    SymbolTable,
    Type as IrType,
    UnitAttr,
    Value,
)


from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from Aipiler.primitive import (
    EinsumPrimitive,
    MapPrimitive,
    ReducePrimitive,
    PopulatePrimitive,
    UnaryPrimitive,
)
from Aipiler.tensor import Tensor
from Aipiler import datatype as dtypes
from iree.compiler import ir
from iree.compiler.dialects.linalg.opdsl.lang import *
from iree.compiler.dialects import builtin
from iree.compiler.dialects import func
from iree.compiler.dialects import linalg
from iree.compiler.dialects import tensor
from Aipiler.aot.support.ir_utils import ModuleBuilder
from Aipiler.graph import EinsumGraph


class Einsum_importer:

    def __init__(
        self,
        module_builder: ModuleBuilder,
    ) -> None:
        self.visited_nodes: List[EinsumPrimitive] = []
        self.symbol_table: Dict[Tensor, ir.Value] = {}
        self.module_builder: ModuleBuilder = module_builder

        self._AIPILER_TO_MLIR: Dict[dtypes.DataType, str] = {
            dtypes.f32: F32Type.get(),
            dtypes.boolean: IntegerType.get_signless(1),
        }

        self._MLIR_TO_AIPILER: Dict[str, dtypes.DataType] = {
            F32Type.get(): dtypes.f32,
            IntegerType.get_signless(1): dtypes.boolean,
        }

    def from_dtype(self, dtype: dtypes.DataType):
        if dtype not in self._AIPILER_TO_MLIR:
            raise RuntimeError("Unsupported data type: {} now".format(dtype.name))
        return self._AIPILER_TO_MLIR[dtype]

    def to_dtype(self, mlirty: str) -> dtypes.DataType:
        if mlirty not in self._MLIR_TO_AIPILER:
            raise RuntimeError("Unsupported data type: {} now".format(mlirty))
        return self._MLIR_TO_AIPILER[mlirty]

    def import_MapPrimitive(
        self,
        node: MapPrimitive,
    ) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        first_value = self.symbol_table[input_tensors[0]]
        second_value = self.symbol_table[input_tensors[1]]
        if (
            first_value not in self.symbol_table
            or second_value not in self.symbol_table
        ):
            raise ValueError(
                f"Input tensor {first_value} and {second_value} not found in symbol table."
            )

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_scripts:
            symbol_defs[script] = getattr(S, script)
            domain_defs[script] = getattr(D, script)

        @linalg_structured_op
        def _map(
            A=TensorDef(T, *(symbol_defs[s] for s in node.lhs_scripts)),
            B=TensorDef(T, *(symbol_defs[s] for s in node.rhs_scripts)),
            C=TensorDef(
                T,
                *(symbol_defs[s] for s in node.output_scripts),
                output=True,
            ),
        ):
            domain((domain_defs[s] for s in node.iteration_scripts))
            output_indices = tuple(domain_defs[s] for s in node.output_scripts)
            lhs_indices = tuple(domain_defs[s] for s in node.lhs_scripts)
            rhs_indices = tuple(domain_defs[s] for s in node.rhs_scripts)
            # TODO: 当前只支持加减乘数,不能写死
            C[output_indices] = A[lhs_indices] * B[rhs_indices]

        mlir_dtype = self.from_dtype(node.output.dtype)
        # TODO: 需要处理symbolic shape
        init_result = tensor.EmptyOp(node.output.shape, mlir_dtype)
        op = _map(
            first_value,
            second_value,
            outs=[init_result.result],
        )

        return op

    def import_ReducePrimitive(self, node: ReducePrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        input_value = self.symbol_table[input_tensors[0]]
        if input_value not in self.symbol_table:
            raise ValueError(f"Input tensor {input_value} not found in symbol table.")

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_scripts:
            symbol_defs[script] = getattr(S, script)
            domain_defs[script] = getattr(D, script)

        @linalg_structured_op
        def _map(
            INPUT=TensorDef(T, *(symbol_defs[s] for s in node.x_scripts)),
            OUTPUT=TensorDef(
                T,
                *(symbol_defs[s] for s in node.output_scripts),
                output=True,
            ),
        ):
            domain(*(domain_defs[s] for s in node.iteration_scripts))
            output_indices = tuple(domain_defs[s] for s in node.output_scripts)
            input_indices = tuple(domain_defs[s] for s in node.x_scripts)
            # TODO: 当前只支持加减乘数,不能写死
            OUTPUT[output_indices] += INPUT[input_indices]

        mlir_dtype = self.from_dtype(node.output.dtype)
        # TODO: 需要处理symbolic shape
        init_result = tensor.EmptyOp(node.output.shape, mlir_dtype)
        op = _map(
            input_value,
            outs=[init_result.result],
        )

        return op

    def import_PopulatePrimitive(self, node: PopulatePrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def import_UnaryPrimitive(self, node: UnaryPrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def import_program(
        self,
        graph: EinsumGraph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:

        # 遍历所有输入张量，生成对应的 MLIR Tensor
        for input_tensor in graph.inputs:
            pass

        with self.module_builder.ip:
            arguments = []
            for inpur_tensor in graph.inputs:
                # TODO：暂时使用tensor的shape，后期再使用symbolic_shape
                dyn = ShapedType.get_dynamic_size()
                shape_list = (dyn,) * len(inpur_tensor.symbolic_shape)
                mlir_dtype = self.from_dtype(input_tensor.dtype)
                tensor_arg = RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)

            @func.FuncOp.from_py_func(*arguments, name=func_name)
            def generated_func(*args):
                # 建立输入参数的符号表
                args_list = list(args)
                for i, arg in enumerate(args_list):
                    self.symbol_table[graph.inputs[i]] = arg

                # 遍历所有节点，生成对应的 MLIR 操作
                for node in graph.nodes:
                    if isinstance(node, MapPrimitive):
                        op_ret: ir.Operation | ir.Value | tuple | List | ir.OpResult = (
                            self.import_MapPrimitive(node)
                        )
                    elif isinstance(node, ReducePrimitive):
                        op_ret: ir.Operation | ir.Value | tuple | List | ir.OpResult = (
                            self.import_ReducePrimitive(node)
                        )
                    else:
                        pass

                    if isinstance(op_ret, tuple | List):
                        for i, operation in enumerate(op_ret):
                            if isinstance(operation, ir.Operation) or isinstance(
                                operation, ir.OpView
                            ):
                                self.symbol_table[node.output] = operation.result
                            elif isinstance(operation, ir.OpResult):
                                self.symbol_table[node.output] = operation
                            else:
                                raise NotImplementedError
                    elif isinstance(op_ret, ir.OpResult):
                        self.symbol_table[node.output] = op_ret
                    else:
                        for i, result in enumerate(op_ret.results):
                            self.symbol_table[node.output] = result

                # 获得函数的所有输出
                outputs = (self.symbol_table.get(out) for out in graph.outputs)
                return outputs

            return generated_func.func_op


class FxImporter:
    """Main entry-point for importing an fx.GraphModule.

    The FxImporter is a low-level class intended for framework integrators.
    It provides several options for customization:

    * config_check: Optionally allows some per-import configuration safety
      checks to be skipped.
    * literal_resolver_callback: Callback that will be invoked when a literal,
      live torch.Tensor is encountered in the FX graph, allowing the default
      action (which is to inline the data as a DenseResourceElementsAttr) to
      be completely overriden.
    * py_attr_tracker: Weak reference tracker for live PyTorch objects used
      to unique them with respect to attributes. If not specified, there will
      be one reference tracker per import, but this can be injected to share
      the same uniqueing across imports (i.e. if building multiple functions
      into the same context or module).
    """

    __slots__ = [
        "_c",
        "_cc",
        "_m",
        "_m_ip",
        "_py_attr_tracker",
        "_hooks",
        "symbol_table",
    ]

    def __init__(
        self,
        *,
        module: Optional[Module] = None,
        context: Optional[Context] = None,
        config_check: bool = True,
        py_attr_tracker: Optional["RefTracker"] = None,
        hooks: Optional[FxImporterHooks] = None,
    ):
        if module is not None:
            assert context is None, "If configuring with a Module, context must be None"
            self._m = module
            self._c = self.module.context
        else:
            self._c = context if context else Context()
            self._m = Module.create(Location.unknown(self._c))
        if config_check:
            # Production code can disable this for a bit of a boost.
            self._config_check()
        self._py_attr_tracker = py_attr_tracker or RefTracker()
        self._cc = ContextCache(self._c, py_attr_tracker=self._py_attr_tracker)
        self._m_ip = InsertionPoint(self._m.body)
        self._hooks = hooks or FxImporterHooks()
        self.symbol_table = SymbolTable(self._m.operation)
        self._hooks.prepare_module(self._m.operation)

    def _config_check(self):
        for dname in REQUIRED_DIALCTS:
            try:
                self._c.dialects[dname]
                logging.debug("Context has registered dialect '%s'", dname)
            except IndexError:
                raise RuntimeError(
                    f"The MLIR context {self._c} is missing required dialect '{dname}'"
                )

    @property
    def module(self) -> Module:
        return self._m

    @property
    def module_op(self) -> Operation:
        return self._m.operation

    def import_program(
        self,
        prog: torch.export.ExportedProgram,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Imports an ExportedProgram according to our chosen canonical representation.

        This mechanism is the fully general solution for handling an ExportedProgram
        and should eventually supercede all others. However, it depends on the
        PyTorch 2.3 release to function properly (specifically, this patch
        made ExportedProgram minimally correct for mutation:
        https://github.com/pytorch/pytorch/pull/118969).

        For stateless programs, the result of this import is a normal function
        defined for immutable `!torch.vtensors`.

        However, if the program mutates its inputs or buffers, then it will be imported
        with those parameters as `!torch.tensor` and appropriate copies and overwrites
        will be done on the inside. Note that the function is still mostly stateless,
        but with `torch.copy.to_vtensor` and `torch.overwrite.tensor.contents`
        ops at the earliest consumer or latest producer to update an argument or
        buffer.

        It is recommended that integrators subclass and override the `resolve_literal`
        method to control access to mutable buffers and parameters. Without that, the
        default policy is to capture them as frozen values.
        """
        # Create lookaside table of placeholders/outputs.
        placeholder_nodes: Dict[str, Node] = {}
        all_producer_nodes: Dict[str, Node] = {}
        loc: Optional[Location] = None
        for node in prog.graph.nodes:
            if loc is None:
                loc = self._cc.get_node_location(node)
            if node.op == "placeholder":
                placeholder_nodes[node.name] = node
                all_producer_nodes[node.name] = node
            elif node.op == "call_function":
                all_producer_nodes[node.name] = node
        if loc is None:
            loc = Location.unknown(self._c)

        # This API is fast evolving. We keep these imports local for now so that we
        # can disable this entire function if needed.
        from torch.export.graph_signature import (
            InputKind,
            OutputKind,
            TensorArgument,
            SymIntArgument,
        )

        sig = prog.graph_signature

        # Populate symbolic guards for dynamic shapes (if any)
        if import_symbolic_shape_expressions:
            self._cc.set_symbolic_guards(prog)

        # Invert the (producer, node_name) maps for mutated user inputs and mutated
        # buffers. This is because we hit-detect based on the input node name.
        mutated_user_inputs = {
            node_name: producer
            for producer, node_name in sig.user_inputs_to_mutate.items()
        }

        # Additional bindings that we need to set up after the function is created.
        mutable_buffer_target_producers: Dict[str, str] = {}
        constant_tensors: Dict[Node, torch.Tensor] = {}
        parameter_bindings: Dict[Node, Tuple[Any, InputInfo]] = {}
        buffer_bindings: Dict[Node, Tuple[Any, InputInfo]] = {}

        # Derive user outputs that we preserve. These will be nodes of the
        # producer for the output.
        user_outputs: List[Node] = []
        user_output_types: List[IrType] = []
        for output_spec in sig.output_specs:
            kind = output_spec.kind
            arg = output_spec.arg
            if kind == OutputKind.USER_OUTPUT:
                if not isinstance(arg, (TensorArgument, SymIntArgument)):
                    raise NotImplementedError(
                        f"OutputKind.USER_OUTPUT for {type(arg)}: {arg}"
                    )
                output_producer_node = all_producer_nodes[arg.name]
                user_outputs.append(output_producer_node)
                user_output_types.append(
                    self._cc.node_val_to_type(output_producer_node)
                )
            elif kind == OutputKind.BUFFER_MUTATION and isinstance(arg, TensorArgument):
                mutable_buffer_target_producers[output_spec.target] = arg.name

        # Derive user inputs. These will be op=='placeholder' nodes.
        user_inputs: List[Node] = []
        user_input_types: List[IrType] = []
        for input_spec in sig.input_specs:
            arg = input_spec.arg
            if input_spec.kind == InputKind.USER_INPUT:
                # Set up user input.
                if not isinstance(arg, (TensorArgument, SymIntArgument)):
                    raise NotImplementedError(
                        f"InputKind.USER_INPUT for {type(arg)}: {arg}"
                    )
                placeholder_node = placeholder_nodes[arg.name]
                mutable = placeholder_node.name in mutated_user_inputs
                user_inputs.append(placeholder_node)
                user_input_types.append(
                    self._cc.node_val_to_type(placeholder_node, mutable=mutable)
                )
            elif input_spec.kind == InputKind.CONSTANT_TENSOR and isinstance(
                arg, TensorArgument
            ):
                # Remember constant tensor binding.
                constant_tensors[placeholder_nodes[arg.name]] = prog.constants[
                    input_spec.target
                ]
            elif input_spec.kind == InputKind.PARAMETER and isinstance(
                arg, TensorArgument
            ):
                # Remember parameter binding.
                value = prog.state_dict.get(input_spec.target)
                assert (
                    not input_spec.persistent or value is not None
                ), "Expected state_dict value for persistent value"
                node = placeholder_nodes[arg.name]
                node_ir_type = self._cc.node_val_to_type(node, mutable=False)
                parameter_bindings[node] = (
                    value,
                    InputInfo(
                        prog,
                        input_spec,
                        node=node,
                        ir_type=node_ir_type,
                        mutable_producer_node_name=None,
                    ),
                )
            elif input_spec.kind == InputKind.BUFFER and isinstance(
                arg, TensorArgument
            ):
                # Remember buffer binding. Unlike user input mutations, buffers
                # are assumed to be represented with load/store semantics based
                # on a symbolic or other non-SSA association. As such, they
                # are not modeled with mutable IR but will trigger an output
                # store hook when the final value is produced.
                if input_spec.persistent:
                    value = prog.state_dict.get(input_spec.target)
                    assert (
                        value is not None
                    ), "Expected state_dict value for persistent buffer"
                else:
                    value = prog.constants.get(input_spec.target)
                    assert (
                        value is not None
                    ), "Expected constants value for non-persistent buffer"

                node = placeholder_nodes[arg.name]
                mutable_producer_node_name = mutable_buffer_target_producers.get(
                    input_spec.target
                )
                node_ir_type = self._cc.node_val_to_type(node, mutable=False)
                buffer_bindings[node] = (
                    value,
                    InputInfo(
                        prog,
                        input_spec,
                        node=node,
                        ir_type=node_ir_type,
                        store_producer_node=mutable_producer_node_name,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"InputSpec not of a known kind: {input_spec}"
                )

        ftype = FunctionType.get(user_input_types, user_output_types, context=self._c)

        # Create the function.
        with loc:
            func_op = func_dialect.FuncOp(
                func_name, ftype, ip=self._m_ip, visibility=func_visibility
            )
            # Programs imported from FX have strong guarantees. Setting this attribute
            # causes various lowerings to be able to emit more efficient code or
            # handle more cases. See isAssumingStrictSymbolicShapes().
            func_op.attributes["torch.assume_strict_symbolic_shapes"] = UnitAttr.get()
            entry_block = Block.create_at_start(func_op.body, ftype.inputs)

        node_importer = GraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
        )

        # Bind constants to IR values.
        for constant_node, constant_tensor in constant_tensors.items():
            node_importer.import_constant(loc, constant_node, constant_tensor)

        # Bind user inputs to IR values.
        for user_input_node, block_arg_value in zip(user_inputs, entry_block.arguments):
            if user_input_node.name in mutated_user_inputs:
                # Materialize
                node_importer.import_mutable_to_vtensor(
                    loc,
                    user_input_node,
                    block_arg_value,
                    mutated_user_inputs[user_input_node.name],
                )
            else:
                # Normal value tensor binding.
                node_importer.bind_node_value(user_input_node, block_arg_value)

        # Lazy bind buffer and parameter inputs.
        for node, (parameter_value, info) in parameter_bindings.items():
            node_importer.lazy_import_parameter(loc, node, parameter_value, info)
        for node, (buffer_value, info) in buffer_bindings.items():
            node_importer.lazy_import_buffer(loc, node, buffer_value, info)

        # Import all nodes and return.
        node_importer.import_nodes(
            all_producer_nodes.values(),
            skip_placeholders_outputs=True,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )
        node_importer.return_node_values(loc, user_outputs)
        self.symbol_table.insert(func_op)
        return func_op

    def import_frozen_program(
        self,
        prog: torch.export.ExportedProgram,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Imports a consolidated torch.export.ExportedProgram instance.

        If using the new torch.export path (vs a lower level precursor), then this is
        the recommended way to canonically use this importer.

        The ExportedProgram form differs from some of the earlier work primarily in
        how it deals with references to external tensors from "outside". In this form,
        all such references are checked to have originated from within the exported
        scope or from an @assume_constant_result wrapped function. Then they are
        transformed to graph inputs and stashed in one of two data structures on
        the ExportedProgram:
        inputs_to_buffers / buffers : For non-parameter buffers.
        inputs_to_parameters / parameters : For parameter buffers.
        The values of the mapping in inputs_to_{buffers|parameters} are in the
        state_dict. This replaces get_attr nodes that would have classically been
        present during lower level tracing.
        Historically, torch-mlir has assumed that all such external accesses are
        frozen, and this entry-point preserves this behavior, treating each distinct
        torch.Tensor encountered in such a way as a `torch.vtensor.literal` (or
        delegating to the literal_resolver_callback to make a policy decision).

        As we anticipate more nuanced treatment options in the future, we name this
        method to indicate that it is producing "frozen" modules. Additional top-level
        approaches to handling state can be introduced later as an addition.

        TODO: This mechanism should be eventually replaced by `import_program` with
        hooks set on the subclass to freeze parameters and buffers. However, that is
        waiting for the Torch 2.3 release cut.
        """
        sig = prog.graph_signature
        state_dict = prog.state_dict
        arg_replacements: Dict[str, Any] = {}

        # Populate symbolic guards for dynamic shapes (if any)
        if import_symbolic_shape_expressions:
            self._cc.set_symbolic_guards(prog)

        # If there is no "constants" attribute, consult the "state_dict". Otherwise, only look
        # at "constants". Relevant upstream patch: https://github.com/pytorch/pytorch/pull/118969
        if hasattr(prog, "constants"):
            constants = prog.constants
            # Lift tensor constants.
            for input_name, state_name in sig.inputs_to_lifted_tensor_constants.items():
                try:
                    state_value = constants[state_name]
                except KeyError as e:
                    raise AssertionError(
                        "Could not find state mapping for tensor constants"
                    ) from e
                arg_replacements[input_name] = state_value
        else:
            # Lift buffers.
            for input_name, state_name in sig.inputs_to_buffers.items():
                try:
                    state_value = state_dict[state_name]
                except KeyError as e:
                    raise AssertionError(
                        "Could not find state mapping for buffer"
                    ) from e
                arg_replacements[input_name] = state_value

        # Lift parameters.
        for input_name, state_name in sig.inputs_to_parameters.items():
            try:
                state_value = state_dict[state_name]
            except KeyError as e:
                raise AssertionError(
                    "Could not find state mapping for parameter"
                ) from e
            arg_replacements[input_name] = state_value

        # Remove any lifted placeholders, replacing their uses with the state
        # replacement value.
        g = prog.graph
        for node in g.nodes:
            if node.op == "placeholder":
                replacement = arg_replacements.get(node.name)
                if replacement is None:
                    continue
                node.replace_all_uses_with(replacement)
                g.erase_node(node)

        return self.import_stateless_graph(
            g,
            func_name=func_name,
            func_visibility=func_visibility,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )

    def import_graph_module(self, gm: GraphModule) -> Operation:
        """Low-level import of a GraphModule assuming that it has been functionalized.

        TODO: This mechanism is deprecated by the `import_program` entry-point and
        it should be removed when no longer required for backwards compatibility.
        """
        return self.import_stateless_graph(gm.graph)

    def import_stateless_graph(
        self,
        g: Graph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Low-level import of a functionalized, assumed stateless Graph as a func.

        TODO: This mechanism is deprecated by the `import_program` entry-point and
        it should be removed when no longer required for backwards compatibility.
        """
        ftype, loc = self._graph_to_function_meta(g)
        # TODO: The FuncOp constructor requires a context-manager context.
        # Fix upstream and then unnest.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/138
        with loc:
            func = func_dialect.FuncOp(
                func_name,
                ftype,
                ip=self._m_ip,
                visibility=func_visibility,
            )
            entry_block = Block.create_at_start(func.body, ftype.inputs)
        node_importer = GraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
        )
        node_importer.import_nodes(
            g.nodes, import_symbolic_shape_expressions=import_symbolic_shape_expressions
        )
        self.symbol_table.insert(func)
        return func

    def _graph_to_function_meta(self, g: Graph) -> Tuple[FunctionType, Location]:
        """Extracts function metadata from the Graph.

        Principally, this includes the FunctionType, but in the future,
        it should also return other annotations (input strides, etc) that
        affect compilation and should be included as arg attrs.
        """
        input_types = []
        result_types = []
        loc = None
        for node in g.nodes:
            # Assume that the first node we can get a location for is about as
            # good as it gets as an overall function location.
            if loc is None:
                loc = self._cc.get_node_location(node)
            if node.op == "placeholder":
                input_types.append(self._cc.node_val_to_type(node))
            elif node.op == "output":
                # An output node's args[0] is the return value. This seems to
                # always be "boxed" as a tuple, which we emit as multi-results.
                for result_node in node.args[0]:
                    if result_node is None:
                        result_types.append(
                            IrType.parse("!torch.none", context=self._c)
                        )
                    elif isinstance(result_node, torch.Tensor):
                        result_types.append(
                            self._cc.tensor_to_vtensor_type(result_node)
                        )
                    elif type(result_node) in SCALAR_TYPE_TO_TORCH_MLIR_TYPE:
                        result_types.append(
                            IrType.parse(
                                SCALAR_TYPE_TO_TORCH_MLIR_TYPE[type(result_node)],
                                self._c,
                            )
                        )
                    else:
                        result_types.append(self._cc.node_val_to_type(result_node))
        return (
            FunctionType.get(input_types, result_types, context=self._c),
            loc if loc else Location.unknown(self._c),
        )
