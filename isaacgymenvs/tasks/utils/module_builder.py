import importlib


def build_module(path_str, build_cfg):
    module = importlib.import_module(path_str)
    function_builder = getattr(module, "build")
    return function_builder(**build_cfg)
