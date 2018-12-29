/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package graph_data_parser;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

public class HDFSWriter extends DataWriter {
  private FileSystem fs;
  private FSDataOutputStream writer;

  public HDFSWriter(String hdfsAddr, String outputpath) throws URISyntaxException, IOException {
    URI uri = new URI(hdfsAddr);
    Configuration con = new Configuration();
    con.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
    fs = FileSystem.get(uri, con);

    Path dfs = new Path(outputpath);
    writer = fs.create(dfs, true);
  }

  @Override
  public void writeFloat(float value) throws IOException {
    writer.write(Bytes.changeBytes(Bytes.floatToBytes(value)));
  }

  @Override
  public void writeInt(int value) throws IOException {
    writer.write(Bytes.changeBytes(Bytes.intToBytes(value)));
  }

  @Override
  public void writeLong(long value) throws IOException {
    writer.write(Bytes.changeBytes(Bytes.longToBytes(value)));
  }

  @Override
  public void writeString(String value) throws IOException {
    writer.write(value.getBytes());
  }

  public void writeFloatList(List<Float> values) throws IOException {
    for (Float value : values) {
      writeFloat(value);
    }
  }

  public void writeLongList(List<Long> values) throws IOException {
    for (Long value : values) {
      writeLong(value);
    }
  }

  public void writeIntList(List<Integer> values) throws IOException {
    for (Integer value : values) {
      writeInt(value);
    }
  }

  public void writeStringList(List<String> values) throws IOException {
    for (String value : values) {
      writeString(value);
    }
  }

  public void writeBytes(byte[] bytes) throws IOException {
    writer.write(bytes);
  }

  public void flush() throws Exception {
    writer.flush();
  }
}
