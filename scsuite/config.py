from . import instyaml


def load(config_dir='./'):
    return instyaml.parse_yaml(open('%s/config.yaml' % config_dir))
