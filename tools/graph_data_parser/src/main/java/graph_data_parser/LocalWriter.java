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

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class LocalWriter extends DataWriter {

  private DataOutputStream writer;

  public LocalWriter(String file) throws Exception {
    writer = new DataOutputStream(
            new BufferedOutputStream(new FileOutputStream(file)));
  }

  public void writeFloat(float value) throws IOException {
    writer.writeFloat(Bytes.bytesToFloat(Bytes.changeBytes(Bytes.floatToBytes(value))));
  }

  public void writeInt(int value) throws IOException {
    writer.writeInt(Bytes.bytesToInt(Bytes.changeBytes(Bytes.intToBytes(value))));
  }

  public void writeLong(long value) throws IOException {
    writer.writeLong(Bytes.bytesToLong(Bytes.changeBytes(Bytes.longToBytes(value))));
  }

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
