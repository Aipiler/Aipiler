def initialize(*args, **kwargs):
    def decorator(f):
        f(*args, **kwargs)

    return decorator
