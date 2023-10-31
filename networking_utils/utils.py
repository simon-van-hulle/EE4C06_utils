import inspect
import numpy as np


def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def LOG(*strings, separator=" "):
    message = separator.join(["[LOG]:", *strings])
    print(message)


def LOGVAR(var, format="10.5f", space=20):
    name = NAME(var)[0]
    extra = "\n" if type(var) == np.ndarray else ""
    try:
        print(f"\t {name:{space}s} : {extra}{var:{format}}")
    except TypeError:
        print(f"\t {name:{space}s} : {extra}{var}")


LV = LOGVAR
