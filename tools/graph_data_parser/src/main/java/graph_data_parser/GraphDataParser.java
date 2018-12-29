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

import com.alibaba.fastjson.JSON;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

class Bytes {
  public static byte[] changeBytes(byte[] a) {
    byte[] b = new byte[a.length];
    for (int i = 0; i < b.length; i++) {
      b[i] = a[b.length - i - 1];
    }
    return b;
  }

  public static int bytesToInt(byte[] bytes) {
    return ByteBuffer.wrap(bytes).getInt();
  }

  public static byte[] intToBytes(int value) {
    return ByteBuffer.allocate(4).putInt(value).array();
  }

  public static byte[] floatToBytes(float value) {
    return ByteBuffer.allocate(4).putFloat(value).array();
  }

  public static float bytesToFloat(byte[] bytes) {
    return ByteBuffer.wrap(bytes).getFloat();
  }

  public static byte[] longToBytes(long value) {
    return ByteBuffer.allocate(8).putLong(value).array();
  }

  public static long bytesToLong(byte[] bytes) {
    return ByteBuffer.wrap(bytes).getLong();
  }
}

class LineBuffer {
  DataInputStream reader;
  String line = "";
  int ptr = 0;

  public LineBuffer(DataInputStream reader) {
    this.reader = reader;
  }

  public char getChar() throws IOException {
    if (line != null) {
      while (line != null && line.length() <= ptr) {
        line = reader.readLine();
        ptr = 0;
      }
      if (line != null) {
        return line.charAt(ptr++);
      } else {
        return '\0';
      }
    } else {
      return '\0';
    }
  }
}

public class GraphDataParser {
  private Meta meta;
  private List<DataWriter> writers;
  private LineBuffer lineBuffer;
  private LineBuffer metaLineBuffer;
  private int partitionNum;

  public GraphDataParser(String metaFile, String inputFile, List<String> outputFileList, String hdfsAddr, int partitionNum) throws Exception {
    this.partitionNum = partitionNum;
    writers = new ArrayList<>(partitionNum);
    DataInputStream reader = new DataInputStream(new FileInputStream(inputFile));
    DataInputStream metaReader = new DataInputStream(new FileInputStream(metaFile));
    for (int i = 0; i < outputFileList.size(); ++i) {
      if (hdfsAddr.isEmpty()) {
        System.out.println("local writer");
        writers.add(new LocalWriter(outputFileList.get(i)));
      } else {
        System.out.println("hdfs writer");
        writers.add(new HDFSWriter(hdfsAddr, outputFileList.get(i)));
      }
    }
    lineBuffer = new LineBuffer(reader);
    metaLineBuffer = new LineBuffer(metaReader);
    readMeta();
  }

  public String readBlock() throws IOException {
    StringBuilder result = new StringBuilder();
    // 读第一个{
    char c = '0';
    while (c != '\0' && c != '{') {
      c = lineBuffer.getChar();
    }
    if (c == '{') {
      result.append(c);
      int cnt = 1;
      while (cnt != 0) {
        c = lineBuffer.getChar();
        result.append(c);
        if (c == '{') {
          ++cnt;
        } else if (c == '}') {
          --cnt;
        }
      }
    }
    return result.toString();
  }

  public void readMeta() throws IOException {
    StringBuilder result = new StringBuilder();
    // 读第一个{
    char c = '0';
    while (c != '\0' && c != '{') {
      c = metaLineBuffer.getChar();
    }
    if (c == '{') {
      result.append(c);
      int cnt = 1;
      while (cnt != 0) {
        c = metaLineBuffer.getChar();
        result.append(c);
        if (c == '{') {
          ++cnt;
        } else if (c == '}') {
          --cnt;
        }
      }
    }
    meta = JSON.parseObject(result.toString(), Meta.class);
  }

  public Block parseBlock(String blockStr) {
    return JSON.parseObject(blockStr, Block.class);
  }

  public void outputBlock(Block block, int partitionId) throws Exception {
    writers.get(partitionId).writeBytes(new BlockParser(meta).BlockJsonToBytes(block));
    writers.get(partitionId).flush();
  }

  public void processFile() throws Exception {
    String blockStr = readBlock();
    while (!blockStr.isEmpty()) {
      Block block = parseBlock(blockStr);
      long partitionIdx = block.getNode_id() % partitionNum;
      outputBlock(block, (int) partitionIdx);
      blockStr = readBlock();
    }
  }

  public static void main(String[] args) {
    int partitionNum = 1;
    String inputDir = "./";
    String outputDir = "";
    String hdfsAddr = "";

    partitionNum = Integer.parseInt(args[0]);
    inputDir = args[1];  // some thing like: ./
    outputDir = args[2];  // some thing like: /data/
    hdfsAddr = args[3];  // some thing like: hdfs://localhost:9000/
    // empty hdfsAddr lead to outputing data to local file system

    try {
      List<String> outputFileList = new ArrayList<>(); // sorted by partition id
      for (int i = 0; i < partitionNum; ++i) {
        outputFileList.add(outputDir + i + ".dat");
      }
      GraphDataParser gdp = new GraphDataParser(inputDir + "/meta.txt", inputDir + "/test.txt", outputFileList,
              hdfsAddr, partitionNum);
      gdp.processFile();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}


