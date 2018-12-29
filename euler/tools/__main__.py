# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getopt
import sys

from euler.tools import json2dat


def usage():
  print("""
        usage: python euler.tools -c [input meta data file] -i [input file name] -o [output file name]
        """)


if __name__ == '__main__':
  opts, args = getopt.getopt(sys.argv[1:], "hc:i:o:")
  if len(opts) < 3:
    usage()
    sys.exit()
  for op, value in opts:
    if op == "-c":
      meta_file = value
    if op == "-i":
      input_file = value
    elif op == "-o":
      output_file = value
    elif op == "-h":
      usage()
      sys.exit()
  c = json2dat.Converter(meta_file, input_file, output_file)
  c.do()
