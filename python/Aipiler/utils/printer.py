import contextlib
import textwrap


class TableSection:
    """辅助类，用于在 table 上下文块中收集结构化的行数据."""

    def __init__(self):
        self.rows = []

    def add_row(self, *cells):
        self.rows.append([str(c) for c in cells])


class Printer:
    # 1. 新增一个类属性，用于映射对齐字符串到相应的格式化方法
    ALIGN_MAP = {
        "left": str.ljust,
        "l": str.ljust,
        "center": str.center,
        "c": str.center,
        "right": str.rjust,
        "r": str.rjust,
    }

    def __init__(self, indent_char="    "):
        self.indent_level = 0
        self.indent_char = indent_char
        self._buffer = []

    # ... (get_string, __str__, clear, add_line, _wrap_text 方法保持不变) ...
    def get_string(self, *, clear: bool = True):
        res = "\n".join(self._buffer)
        if clear:
            self._buffer = []
        return res

    def __str__(self):
        return self.get_string()

    def clear(self):
        self._buffer = []

    def add_line(self, text=""):
        if text:
            indentation = self.indent_char * self.indent_level
            self._buffer.append(f"{indentation}{text}")
        else:
            self._buffer.append("")

    @contextlib.contextmanager
    def section(self, head: str = ""):
        self.add_line(head)
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1

    @staticmethod
    def _wrap_text(text: str, width: int) -> list[str]:
        if not text:
            return [""]
        return (
            textwrap.wrap(text, width=width, break_long_words=True)
            if width > 0
            else [text]
        )

    @contextlib.contextmanager
    # 2. 为 table 方法添加 aligns 参数
    def table(
        self,
        separator="  ",
        col_widths: list[int] | None = None,
        aligns: list[str] | None = None,
    ):
        """
        一个上下文管理器，用于接收结构化数据并构建为对齐的表格字符串。
        新增功能：支持通过 aligns 参数指定每列的对齐方式 ('left', 'center', 'right')。
        """
        table_buffer = TableSection()
        yield table_buffer

        rows = table_buffer.rows
        if not rows:
            return

        current_indent = self.indent_char * self.indent_level

        if col_widths is None:
            num_columns = max(len(row) for row in rows) if rows else 0
            max_widths = [0] * num_columns
            for row in rows:
                for i, cell in enumerate(row):
                    if len(cell) > max_widths[i]:
                        max_widths[i] = len(cell)

            for row in rows:
                formatted_cells = []
                # 3. 在格式化单元格时应用指定的对齐方式
                for i in range(num_columns):
                    cell = row[i] if i < len(row) else ""
                    # 获取对齐方式，如果未提供或超出范围，则默认为'left'
                    align = aligns[i].lower() if aligns and i < len(aligns) else "left"
                    # 从ALIGN_MAP获取格式化函数，如果找不到也默认为左对齐
                    formatter = self.ALIGN_MAP.get(align, str.ljust)
                    formatted_cells.append(formatter(cell, max_widths[i]))

                formatted_line = separator.join(formatted_cells)
                self._buffer.append(f"{current_indent}{formatted_line}")
        else:
            num_columns = len(col_widths)
            for row in rows:
                wrapped_cells = []
                for i in range(num_columns):
                    cell_text = row[i] if i < len(row) else ""
                    width = col_widths[i]
                    wrapped_lines = self._wrap_text(cell_text, width)
                    wrapped_cells.append(wrapped_lines)

                num_physical_rows = (
                    max(len(wc) for wc in wrapped_cells) if wrapped_cells else 0
                )

                for i in range(num_physical_rows):
                    physical_row_parts = []
                    for j in range(num_columns):
                        part = wrapped_cells[j][i] if i < len(wrapped_cells[j]) else ""
                        physical_row_parts.append(part)

                    # 4. 在格式化折行的单元格时也应用指定的对齐方式
                    formatted_cells = []
                    for j, part in enumerate(physical_row_parts):
                        width = col_widths[j]
                        align = (
                            aligns[j].lower() if aligns and j < len(aligns) else "left"
                        )
                        formatter = self.ALIGN_MAP.get(align, str.ljust)
                        formatted_cells.append(formatter(part, width))

                    formatted_line = separator.join(formatted_cells)
                    self._buffer.append(f"{current_indent}{formatted_line}")


P = Printer()
