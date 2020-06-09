# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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

import euler
import sys
import time


if __name__ == '__main__':
    shard_idx = sys.argv[1]
    shard_num = sys.argv[2]

    euler.start(directory='/tmp/ppi_data',
                shard_idx=shard_idx,
                shard_num=shard_num,
                zk_addr='127.0.0.1:2181',
                zk_path='/euler-2.0ppi-test',
                global_sampler_type='all',
                server_thread_num=4)

    time.sleep(100000)
