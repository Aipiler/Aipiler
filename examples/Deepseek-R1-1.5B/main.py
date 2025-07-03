import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from torch.export import Dim, export
import json
import iree.runtime as rt
from iree.turbine import aot
import numpy as np
from Aipiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner


def load_deepseek_model(local_model_path):
    """加载DeepSeek模型和tokenizer"""
    print("🚀 加载DeepSeek模型...")

    # 加载配置
    config = AutoConfig.from_pretrained(
        local_model_path, local_files_only=True)
    print(f"模型配置: {config.model_type}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16   # 在这里指定权重数据类型为bfloat16
    )

    # 设置为评估模式
    model.eval()

    print(f"✅ 模型加载完成")
    print(f"   - 模型类型: {type(model).__name__}")
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 参数类型: {model.parameters().__next__().dtype}")
    print(f"   - 词汇表大小: {tokenizer.vocab_size}")

    return model, tokenizer, config


def create_example_inputs(tokenizer, config, batch_size=1, seq_length=32):
    """创建模型输入示例 - 为torch.export优化"""
    print(f"📝 创建示例输入 (batch_size={batch_size}, seq_length={seq_length})")

    # torch.export需要确定的输入形状，使用较小的序列长度
    # 使用真实的token编码
    sample_text = "Hello world"
    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        max_length=seq_length,
        truncation=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)

    print(f"   - input_ids形状: {input_ids.shape}")
    print(f"   - input_ids dtype: {input_ids.dtype}")
    if attention_mask is not None:
        print(f"   - attention_mask形状: {attention_mask.shape}")
        print(f"   - attention_mask形状: {attention_mask.dtype}")
    print(f"   - 示例文本: {sample_text}")

    return input_ids, attention_mask


def export_model_with_torch_export(
    model, input_ids, attention_mask=None
) -> torch.export.ExportedProgram:
    """使用torch.export导出模型"""
    print("🔄 使用torch.export导出PyTorch模型...")

    try:
        # 先进行一次前向传播测试
        print("   - 测试前向传播...")
        with torch.no_grad():
            if attention_mask is not None:
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask)
                print(f"   - 输出logits形状: {output.logits.shape}")
            else:
                output = model(input_ids=input_ids)
                print(f"   - 输出logits形状: {output.logits.shape}")

        print("   - 尝试直接导出模型...")
        try:
            if attention_mask is not None:
                example_args = (input_ids, attention_mask)
                print(
                    f"   - 输入参数: input_ids{input_ids.shape}, attention_mask{attention_mask.shape}"
                )
            else:
                example_args = (input_ids,)
                print(f"   - 输入参数: input_ids{input_ids.shape}")

            # Create a dynamic batch size
            batch = Dim("batch")
            seq_len = Dim("seq_len", max=4095)
            dynamic_shapes = {
                "input_ids": {0: batch},
                "input_ids": {1: seq_len},
                "attention_mask": {0: batch},
                "attention_mask": {1: seq_len},
            }
            # 直接传递模型对象和位置参数
            exported_program = export(
                model, args=example_args, dynamic_shapes=dynamic_shapes
            )
            print("✅ 直接导出成功")
            return exported_program

        except Exception as e1:
            print(f"   ❌ 直接导出失败: {e1}")

    except Exception as e:
        print(f"❌ torch.export导出失败: {e}")
        print("详细错误信息:")
        import traceback

        traceback.print_exc()


def analyze_exported_graph(exported_program: torch.export.ExportedProgram):
    """分析导出的图结构"""
    if exported_program is None:
        print("⚠️ 没有导出的模型可供分析")
        return

    print("\n" + "=" * 60)
    print("📊 分析torch.export导出的图结构")
    print("=" * 60)

    # 打印图的基本信息
    print(f"图节点数量: {len(exported_program.graph.nodes)}")

    # 统计节点类型
    node_types = {}
    for node in exported_program.graph.nodes:
        node_type = node.op
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print("\n节点类型统计:")
    for node_type, count in node_types.items():
        print(f"  {node_type:15}: {count:4d} 个")

    # 分析具体节点
    print(f"\n🔍 详细节点分析:")
    print("-" * 60)

    for i, node in enumerate(exported_program.graph.nodes):
        if i >= 20:  # 只显示前20个节点
            print(f"... (还有 {len(exported_program.graph.nodes) - 20} 个节点)")
            break

        print(f"\n节点 {i+1}: {node.name}")
        print(f"  操作类型: {node.op}")

        if node.op == "placeholder":
            print(f"  参数: {node.args}")
            if hasattr(node, "meta") and "val" in node.meta:
                val = node.meta["val"]
                if hasattr(val, "shape"):
                    print(f"  输入形状: {val.shape}")
                    print(f"  数据类型: {val.dtype}")

        elif node.op == "call_function":
            print(f"  调用函数: {node.target}")
            print(f"  参数数量: {len(node.args)}")

            # 打印前几个参数的信息
            for j, arg in enumerate(node.args[:3]):
                if hasattr(arg, "name"):
                    print(f"    参数{j+1}: {arg.name} (节点)")
                else:
                    print(f"    参数{j+1}: {type(arg)} - {str(arg)[:50]}")

            if len(node.args) > 3:
                print(f"    ... (还有 {len(node.args)-3} 个参数)")

            # 输出信息
            if hasattr(node, "meta") and "val" in node.meta:
                val = node.meta["val"]
                if hasattr(val, "shape"):
                    print(f"  输出形状: {val.shape}")
                    print(f"  输出类型: {val.dtype}")
                elif isinstance(val, (list, tuple)):
                    print(f"  输出类型: {type(val)} (长度: {len(val)})")
                else:
                    print(f"  输出: {type(val)}")

        elif node.op == "output":
            print(f"  输出节点")
            print(f"  参数: {node.args}")


def analyze_graph_operators(exported_program):
    """分析图中的算子类型"""
    if exported_program is None:
        return

    print("\n" + "=" * 60)
    print("🔧 算子类型分析")
    print("=" * 60)

    operators = {}

    for node in exported_program.graph.nodes:
        if node.op == "call_function":
            op_name = str(node.target)
            operators[op_name] = operators.get(op_name, 0) + 1
        elif node.op == "call_method":
            method_name = str(node.target)
            operators[f"method_{method_name}"] = (
                operators.get(f"method_{method_name}", 0) + 1
            )

    print("检测到的算子:")
    for op_name, count in sorted(operators.items()):
        print(f"  {op_name:30}: {count:4d} 次")

    return operators


def save_export_analysis(exported_program, output_dir="./export_analysis"):
    """保存导出分析结果"""
    if exported_program is None:
        return

    print(f"\n💾 保存导出分析结果到 {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存图结构
    torch.export.save(exported_program, output_dir + "/exported_program.pt2")
    with open(
        os.path.join(output_dir, "exported_graph.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(str(exported_program))

    # 保存节点详情
    # nodes_info = []
    # for node in exported_program.graph.nodes:
    #     node_info = {
    #         "name": node.name,
    #         "op": node.op,
    #         "target": str(node.target) if node.target else None,
    #         "args_count": len(node.args),
    #     }

    #     if hasattr(node, "meta") and "val" in node.meta:
    #         val = node.meta["val"]
    #         if hasattr(val, "shape"):
    #             node_info["output_shape"] = list(val.shape)
    #             node_info["output_dtype"] = str(val.dtype)

    #     nodes_info.append(node_info)

    # with open(
    #     os.path.join(output_dir, "nodes_analysis.json"), "w", encoding="utf-8"
    # ) as f:
    #     json.dump(nodes_info, f, indent=2, ensure_ascii=False)

    print("✅ 导出分析结果已保存")


def main():
    """主函数"""
    local_model_path = "/home/gsh/DeepSeek-R1-Distill-Qwen-1.5B"

    try:
        # 1. 加载模型
        model, tokenizer, config = load_deepseek_model(local_model_path)

        # 2. 创建示例输入 (使用较小的尺寸以适配torch.export)
        input_ids, attention_mask = create_example_inputs(
            tokenizer, config, batch_size=1, seq_length=32  # 较小的序列长度
        )

        # target_backend = "cuda"
        # device = "cuda"

        # example_args = (input_ids, attention_mask)
        # exported = aot.export(model, args=example_args)
        # # exported.print_readable()
        # compiled_binary = exported.compile(
        #     save_to=None, target_backends=target_backend)

        # config = rt.Config(device)
        # vm_module = rt.VmModule.copy_buffer(
        #     config.vm_instance, compiled_binary.map_memory()
        # )

        # inputs = [input_ids.numpy(), attention_mask.numpy()]
        # benchmark_config = BenchmarkConfig(num_runs=10)
        # benchmarker = BenchmarkRunner(benchmark_config)
        # result = benchmarker.run_benchmark(
        #     vm_module,
        #     "main",
        #     inputs,
        #     f"main",
        #     device=device,
        # )
        # benchmarker.print_result_simple(result)

        # 3. 使用torch.export导出模型
        exported_program = export_model_with_torch_export(
            model, input_ids, attention_mask
        )

        if exported_program is not None:
            # 4. 分析导出的图
            analyze_exported_graph(exported_program)

            # 5. 分析算子类型
            operators = analyze_graph_operators(exported_program)

            # 6. 保存分析结果
            save_export_analysis(exported_program)

            print("\n✅ torch.export导出和分析完成")

        # 7. 额外的模型结构分析
        print("\n" + "=" * 60)
        print("📋 补充模型结构信息")
        print("=" * 60)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")
        print(f"模型类型: {type(model).__name__}")
        print(f"配置信息:")

        key_configs = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
        ]
        for key in key_configs:
            if hasattr(config, key):
                print(f"  {key}: {getattr(config, key)}")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()