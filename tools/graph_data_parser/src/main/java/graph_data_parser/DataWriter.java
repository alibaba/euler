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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataWriter {
  private int bytesNum = 0;
  private List<byte[]> result = new ArrayList<>();

  public void writeFloat(float value) throws IOException {
    byte[] bytes = Bytes.changeBytes(Bytes.floatToBytes(value));
    bytesNum += bytes.length;
    result.add(bytes);
  }

  public void writeInt(int value) throws IOException {
    byte[] bytes = Bytes.changeBytes(Bytes.intToBytes(value));
    bytesNum += bytes.length;
    result.add(bytes);
  }

  public void writeLong(long value) throws IOException {
    byte[] bytes = Bytes.changeBytes(Bytes.longToBytes(value));
    bytesNum += bytes.length;
    result.add(bytes);
  }

  public void writeString(String value) throws IOException {
    byte[] bytes = value.getBytes();
    bytesNum += bytes.length;
    result.add(bytes);
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

  public byte[] getBytes() {
    byte[] bytes = new byte[bytesNum];
    int dstptr = 0;
    for (int i = 0; i < result.size(); ++i) {
      System.arraycopy(result.get(i), 0, bytes, dstptr, result.get(i).length);
      dstptr += result.get(i).length;
    }
    return bytes;
  }

  public void writeBytes(byte[] bytes) throws IOException {
  }

  public void flush() throws Exception {

  }
}
