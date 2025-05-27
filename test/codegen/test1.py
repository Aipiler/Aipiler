from mlir.ir import Context, Module

with Context() as ctx:  # ctx 是 mlir.ir.Context 的一个实例
    # IR construction using `ctx` as context.
    # 在这里进行 IR 构建，`ctx` 会作为（隐式的）上下文

    # For example, parsing an MLIR module from string requires the context.
    # 例如，从字符串解析 MLIR 模块需要上下文。
    Module.parse("builtin.module {}")  # 在 with 块内，parse 会自动使用 ctx
