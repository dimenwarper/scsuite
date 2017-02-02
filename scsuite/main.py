import argparse
from . import commands


def top_level_command():
    COMMANDS = {'start': commands.StartCommand(),
                'split': commands.SplitCommand(),
                'score': commands.ScoreCommand(),
                'recommend-model': commands.ModelRecommendationCommand(),
                'fit': commands.FitCommand(),
                'scran': commands.SCRANCommand(),
                'm3drop': commands.M3DropCommand()}

    parser = argparse.ArgumentParser(prog='CMD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help='sub-command helps')

    for command, handler in COMMANDS.iteritems():
        command_subparser = subparsers.add_parser(command)
        command_subparser.set_defaults(handler=handler)
        command_subparser = handler.setup_clparser(command_subparser)
    clargs = parser.parse_args()
    clargs.handler.execute(clargs)
