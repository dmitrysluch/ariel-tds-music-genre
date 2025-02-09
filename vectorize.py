import numpy as np
import numba as nb
import types
import dis
from types import CodeType
import sys

# Patch method bytecode to write to provided array instead of returning
def patch_bytecode(f):
    code = f.__code__
    constants = list(code.co_consts)
    bytecode = list(dis.get_instructions(f))
    if sum(1 for instr in bytecode if instr.opname == 'RETURN_VALUE') > 1:
        raise NotImplementedError("Can't vectorize function with more then one return statement, yet.")
    res_bytecode = []
    # Create result value arg
    RESULT_ARG_NAME = "__internal_result"
    varnames = code.co_varnames[:code.co_argcount] + (RESULT_ARG_NAME,) + code.co_varnames[code.co_argcount:]
    result_idx = varnames.index(RESULT_ARG_NAME)
    # Replace RETURN_VALUE with STORE_FAST and RETURN_NONE
    for instr in bytecode:
        res_bytecode += [bytes((0, 0))] * max(instr.offset // 2 - len(res_bytecode), 0)
        if instr.opname == 'RETURN_VALUE':
            # Add None to constants if not present
            if None not in constants:
                constants.append(None)
            if ... not in constants:
                constants.append(...)
            none_idx = constants.index(None)
            ellipsis_idx = constants.index(...)
            
            # Replace RETURN_VALUE with STORE_FAST and LOAD_CONST(None)
            res_bytecode.append(bytes((dis.opmap['LOAD_FAST'], result_idx)))
            res_bytecode.append(bytes((dis.opmap['LOAD_CONST'], ellipsis_idx)))
            res_bytecode.append(bytes((dis.opmap['STORE_SUBSCR'], 0, 0, 0)))
            res_bytecode.append(bytes((dis.opmap['LOAD_CONST'], none_idx)))
            res_bytecode.append(bytes((dis.opmap['RETURN_VALUE'], 0)))
            break
        else:
            arg = instr.arg 
            if arg == None:
                arg = 0
            elif instr.opname in ['LOAD_FAST', 'STORE_FAST', 'DELETE_FAST'] and arg >= code.co_argcount:
                arg += 1
            res_bytecode.append(bytes((instr.opcode, arg)))
    
    res_bytecode = b''.join(res_bytecode)
    
    new_code = CodeType(code.co_argcount + 1,
                      code.co_posonlyargcount,
                      code.co_kwonlyargcount,
                      len(varnames),
                      code.co_stacksize,
                      code.co_flags,
                      res_bytecode,
                      tuple(constants),
                      code.co_names,
                      varnames,
                      code.co_filename,
                      code.co_name,
                      code.co_qualname,
                      code.co_firstlineno,
                      code.co_linetable,
                      code.co_exceptiontable,
                      code.co_freevars,
                      code.co_cellvars)
    
    return types.FunctionType(new_code, f.__globals__, f.__name__, f.__defaults__, f.__closure__)

def arr_to_signature(arr: np.ndarray):
    return nb.from_dtype(arr.dtype)[(slice(None, None),) * len(arr.shape)]

def create_vectorized_mapper(method, example_args, example_out):    
    try:
        # Create patched version of the method
        patched = patch_bytecode(method)
        
        # Try to create vectorized version using numba
        sig = tuple(arr_to_signature(arg) for arg in example_args)
        sig = sig + (arr_to_signature(example_out),)
        
        letter = ord('a') + 0
        shape_sign = []
        for arg in example_args:
            arg_sign = []
            for _ in arg.shape:
                arg_sign.append(chr(letter))
                letter += 1
            shape_sign.append(f"({','.join(arg_sign)})")
        arg_sign = []
        for _ in example_out.shape:
            arg_sign.append(chr(letter))
            letter += 1
        shape_sign = ','.join(shape_sign + [f"({','.join(arg_sign)})"])
        print(sig)
        print(shape_sign)
        vectorized = nb.guvectorize([sig], shape_sign, nopython=True)(patched)
        use_numba = True
        print("Using numba compiled wrapper")
    except Exception as e:
        print(f"Failed compiling function, will use plain python wrapper.")
        use_numba = False
    
    def wrapper(*args, result):
        batch_size = args[0].shape[0]
        
        if use_numba:
            vectorized(*(args + (result,)))
        else:
            for i in range(batch_size):
                result[i] = method(*(arg[i] for arg in args))
    
    return wrapper

def patch_bytecode_multi(f, num_outputs):
    code = f.__code__
    constants = list(code.co_consts)
    bytecode = list(dis.get_instructions(f))
    if sum(1 for instr in bytecode if instr.opname == 'RETURN_VALUE') > 1:
        raise NotImplementedError("Can't vectorize function with more then one return statement, yet.")
    
    res_bytecode = []
    i = 0
    for instr in bytecode:
        res_bytecode += [bytes((0, 0))] * max(instr.offset // 2 - len(res_bytecode), 0)
        if instr.opname == 'RETURN_VALUE':
            if None not in constants:
                constants.append(None)
            if ... not in constants:
                constants.append(...)
            for i in range(num_outputs):
                if i not in constants:
                    constants.append(i)
            none_idx = constants.index(None)
            ellipsis_idx = constants.index(...)
            idx_idx = [constants.index(i) for i in range(num_outputs)]
            
            # Store each element of the tuple in corresponding result array
            for i in range(num_outputs):
                res_bytecode.append(bytes((dis.opmap['COPY'], 0)))
                res_bytecode.append(bytes((dis.opmap['LOAD_CONST'], idx_idx[i])))
                res_bytecode.append(bytes((dis.opmap['BINARY_SUBSCR'], 0, 0, 0)))
                res_bytecode.append(bytes((dis.opmap['LOAD_FAST'], len(f.__code__.co_varnames) + i)))
                res_bytecode.append(bytes((dis.opmap['LOAD_CONST'], ellipsis_idx)))
                res_bytecode.append(bytes((dis.opmap['STORE_SUBSCR'], 0, 0, 0)))
            
            res_bytecode.append(bytes((dis.opmap['POP_TOP'], 0)))
            res_bytecode.append(bytes((dis.opmap['LOAD_CONST'], none_idx)))
            res_bytecode.append(bytes((dis.opmap['RETURN_VALUE'], 0)))
            break
        else:
            res_bytecode.append(bytes((instr.opcode, instr.arg if instr.arg is not None else 0)))
    
    res_bytecode = b''.join(res_bytecode)
    
    varnames = code.co_varnames + tuple(f'result_{i}' for i in range(num_outputs))
    
    new_code = CodeType(code.co_argcount + num_outputs,
                      code.co_posonlyargcount,
                      code.co_kwonlyargcount,
                      len(varnames),
                      code.co_stacksize,
                      code.co_flags,
                      res_bytecode,
                      tuple(constants),
                      code.co_names,
                      varnames,
                      code.co_filename,
                      code.co_name,
                      code.co_qualname,
                      code.co_firstlineno,
                      code.co_linetable,
                      code.co_exceptiontable,
                      code.co_freevars,
                      code.co_cellvars)
    
    return types.FunctionType(new_code, f.__globals__, f.__name__, f.__defaults__, f.__closure__)

def create_vectorized_mapper_multi(method, example_args, example_outs):
    try:
        patched = patch_bytecode_multi(method, len(example_outs))
        
        sig = tuple(arr_to_signature(arg) for arg in example_args)
        sig = sig + tuple(arr_to_signature(out) for out in example_outs)
        
        letter = ord('a')
        shape_sign = []
        for arg in example_args + example_outs:
            arg_sign = []
            for _ in arg.shape:
                arg_sign.append(chr(letter))
                letter += 1
            shape_sign.append(f"({','.join(arg_sign)})")
        shape_sign = ','.join(shape_sign)
        
        vectorized = nb.guvectorize([sig], shape_sign, nopython=True)(patched)
        use_numba = True
        print("Using numba compiled wrapper")
    except Exception as e:
        print(f"Failed compiling function, will use plain python wrapper, exception: {e}")
        use_numba = False
    
    def wrapper(*args, result):
        batch_size = args[0].shape[0]
        if use_numba:
            vectorized(*(args + result))
        else:
            for i in range(batch_size):
                outs = method(*(arg[i] for arg in args))
                for j, out in enumerate(outs):
                    result[j][i] = out
        
    
    return wrapper