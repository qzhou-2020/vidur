from dataclasses import fields, is_dataclass
from typing import Union, get_args, get_origin
import json

primitive_types = {int, str, float, bool, type(None)}


def get_all_subclasses(cls):
    subclasses = cls.__subclasses__()
    return subclasses + [g for s in subclasses for g in get_all_subclasses(s)]


def is_primitive_type(field_type: type) -> bool:
    # Check if the type is a primitive type
    return field_type in primitive_types


def is_generic_composed_of_primitives(field_type: type) -> bool:
    origin = get_origin(field_type)
    if origin in {list, dict, tuple, Union}:
        # Check all arguments of the generic type
        args = get_args(field_type)
        return all(is_composed_of_primitives(arg) for arg in args)
    return False


def is_composed_of_primitives(field_type: type) -> bool:
    # Check if the type is a primitive type
    if is_primitive_type(field_type):
        return True

    # Check if the type is a generic type composed of primitives
    if is_generic_composed_of_primitives(field_type):
        return True

    return False


def to_snake_case(name: str) -> str:
    return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip("_")


def is_optional(field_type: type) -> bool:
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def is_list(field_type: type) -> bool:
    # Check if the field type is a List
    return get_origin(field_type) is list


def is_dict(field_type: type) -> bool:
    # Check if the field type is a Dict
    return get_origin(field_type) is dict


def is_bool(field_type: type) -> bool:
    return field_type is bool


def get_inner_type(field_type: type) -> type:
    return next(t for t in get_args(field_type) if t is not type(None))


def is_subclass(cls, parent: type) -> bool:
    return hasattr(cls, "__bases__") and parent in cls.__bases__


def dataclass_to_dict(obj):
    """Convert a dataclass instance to a dict, skipping over non-serializable attributes."""
    if not hasattr(obj, "__dict__"):
        return obj
    
    result = {}
    for key, value in obj.__dict__.items():
        # Skip functions and methods
        if callable(value):
            continue
            
        # Handle nested dataclasses
        if hasattr(value, "__dict__"):
            result[key] = dataclass_to_dict(value)
        # Handle lists/tuples
        elif isinstance(value, (list, tuple)):
            result[key] = [dataclass_to_dict(item) for item in value]
        # Handle dictionaries
        elif isinstance(value, dict):
            result[key] = {k: dataclass_to_dict(v) for k, v in value.items()}
        # Handle basic types
        else:
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                result[key] = value
            except (TypeError, OverflowError):
                # Skip non-serializable values
                continue
                
    return result
