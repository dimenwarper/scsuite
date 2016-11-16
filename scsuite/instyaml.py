from . import utils
import yaml


def _serialize_dict(dic):
    serialized_dict = {}
    for k, v in dic.items():
        if k == 'class':
            serialized_dict[k] = '%s.%s' % (v.__module__, v.__name__)
        elif type(v) == dict:
            serialized_dict[k] = _serialize_dict(v)
        else:
            serialized_dict[k] = v
    return serialized_dict


def _parse_value(key, val):
    if key == 'class':
        return utils.str_to_class(val)
    else:
        if type(val) == dict:
            return _parse_dict(val)
        else:
            return val


def _parse_dict(yaml_dict):
    parsed_dict = {}
    for key, val in yaml_dict.items():
        parsed_dict[key] = _parse_value(key, val)
    return parsed_dict


def parse_yaml(fl):
    yaml_dict = yaml.safe_load(fl)
    return _parse_dict(yaml_dict)


def dump(obj, fl, **kwargs):
    if type(obj) == dict:
        obj = _serialize_dict(obj)
    yaml.dump(obj, fl, **kwargs)
