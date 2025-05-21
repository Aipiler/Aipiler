import inspect
import ast


def my_decorator(func):
    print("装饰器被应用了")  # 这是在函数定义时打印的

    source = inspect.getsource(func)
    tree = ast.parse(source)
    function_def = tree.body[0]
    print(function_def)

    def wrapper():
        print("函数被调用了")
        return func()

    return wrapper


@my_decorator
def say_hello():
    print("你好!")


# 这时还没有打印"函数被调用了"，只是打印"装饰器被应用了"
# say_hello()
