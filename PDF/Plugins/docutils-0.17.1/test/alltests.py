#!/bin/sh
''''exec python -u "$0" "$@" #'''
from __future__ import print_function

# $Id: alltests.py 8346 2019-08-26 12:11:32Z milde $
# Author: David Goodger <goodger@python.org>
# Copyright: This module has been placed in the public domain.

__doc__ = """\
All modules named 'test_*.py' in the current directory, and recursively in
subdirectories (packages) called 'test_*', are loaded and test suites within
are run.
"""

import time
# Start point for actual elapsed time, including imports
# and setup outside of unittest.
start = time.time()  # noqa

import sys
import atexit
import os
import platform
import DocutilsTestSupport              # must be imported before docutils
import docutils


class Tee(object):

    """Write to a file and a stream (default: stdout) simultaneously."""

    def __init__(self, filename, stream=sys.__stdout__):
        self.file = open(filename, 'w')
        atexit.register(self.close)
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', None)

    def close(self):
        self.file.close()
        self.file = None

    def write(self, string):
        try:
            self.stream.write(string)
            if self.file:
                self.file.write(string)
        except UnicodeEncodeError:   # Py3k writing to "ascii" stream/file
            string = string.encode('raw_unicode_escape').decode('ascii')
            self.stream.write(string)
            if self.file:
                self.file.write(string)

    def flush(self):
        self.stream.flush()
        if self.file:
            self.file.flush()


def pformat(suite):
    step = 4
    suitestr = repr(suite).replace('=[<', '=[\n<').replace(', ', ',\n')
    indent = 0
    output = []
    for line in suitestr.splitlines():
        output.append(' ' * indent + line)
        if line[-1:] == '[':
            indent += step
        else:
            if line[-5:] == ']>]>,':
                indent -= step * 2
            elif line[-3:] == ']>,':
                indent -= step
    return '\n'.join(output)


def suite():
    path, script = os.path.split(sys.argv[0])
    suite = package_unittest.loadTestModules(DocutilsTestSupport.testroot,
                                             'test_', packages=1)
    sys.stdout.flush()
    return suite


# must redirect stderr *before* first import of unittest
sys.stdout = sys.stderr = Tee('alltests.out')

import package_unittest  # noqa


if __name__ == '__main__':
    suite = suite()
    print('Testing Docutils %s with Python %s on %s at %s' % (
        docutils.__version__, sys.version.split()[0],
        time.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')))
    print('OS: %s %s %s (%s, %s)' % (
        platform.system(), platform.release(), platform.version(),
        sys.platform, platform.platform()))
    print('Working directory: %s' % os.getcwd())
    print('Docutils package: %s' % os.path.dirname(docutils.__file__))
    sys.stdout.flush()
    result = package_unittest.main(suite)
    finish = time.time()
    print('Elapsed time: %.3f seconds' % (finish - start))
    sys.exit(not result.wasSuccessful())
