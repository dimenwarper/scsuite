from __future__ import print_function
from colorama import Fore, Style
import sys

def info(message):
    print(Fore.GREEN + '[INFO] ', file=sys.stderr, end='')
    print(Style.RESET_ALL, file=sys.stderr, end='')
    print(message, file=sys.stderr)


def warning(message):
    print(Fore.YELLOW + '[WARNING] ', end='', file=sys.stderr)
    print(Style.RESET_ALL, end='', file=sys.stderr)
    print(message, file=sys.stderr)


def error(message):
    print(Fore.RED + '[ERROR] ', end='', file=sys.stderr)
    print(Style.RESET_ALL, end='', file=sys.stderr)
    print(message, file=sys.stderr)


def debug(message):
    print(Fore.BLUE + '[DEBUG] ', end='', file=sys.stderr)
    print(Style.RESET_ALL, end='', file=sys.stderr)
    print(message, file=sys.stderr)


def result(message):
    print(message, file=sys.stdout)
