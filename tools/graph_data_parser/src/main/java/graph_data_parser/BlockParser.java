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

public class BlockParser {
  private Meta meta;

  public BlockParser(Meta m) {
    meta = m;
  }

  public byte[] BlockJsonToBytes(Block block) throws IOException {
    DataWriter writer = new DataWriter();
    int blockBytes = 0;
    int nodeInfoBytes = 0;
    ArrayList<Integer> edgesInfoBytes = new ArrayList<>();

    // get nodeInfoBytes
    nodeInfoBytes = 8 + 4 + 4 +  // node_id + node_type + node_weight
            4 + meta.getEdge_type_num() * 4 + meta.getEdge_type_num() * 4;  // edge group info

    int neighborNum = 0;
    for (int i = 0; i < meta.getEdge_type_num(); ++i) {
      neighborNum += block.getNeighbor().get(i) == null ? 0 :
              block.getNeighbor().get(i).size();
    }
    nodeInfoBytes += neighborNum * 8 + neighborNum * 4;  // neighbor info

    int uint64FeatureNum = 0;
    for (int i = 0; i < meta.getNode_uint64_feature_num(); ++i) {
      uint64FeatureNum += block.getUint64_feature().get(i) == null ? 0 :
              block.getUint64_feature().get(i).size();
    }
    nodeInfoBytes += 4 + meta.getNode_uint64_feature_num() * 4 + uint64FeatureNum * 8;  // ui64 feature info size

    int floatFeatureNum = 0;
    for (int i = 0; i < meta.getNode_float_feature_num(); ++i) {
      floatFeatureNum += block.getFloat_feature().get(i) == null ? 0 :
              block.getFloat_feature().get(i).size();
    }
    nodeInfoBytes += 4 + meta.getNode_float_feature_num() * 4 + floatFeatureNum * 4;  // float feature info size

    int binaryFeatureNum = 0;
    for (int i = 0; i < meta.getNode_binary_feature_num(); ++i) {
      binaryFeatureNum += block.getBinary_feature().get(i) == null ? 0 :
              block.getBinary_feature().get(i).length();
    }
    nodeInfoBytes += 4 + meta.getNode_binary_feature_num() * 4 + binaryFeatureNum;

    // get edgeInfoBytes
    int totalEdgeInfoBytes = 0;
    for (int i = 0; i < block.getEdge().size(); ++i) {
      Edge edge = block.getEdge().get(i);
      int edgeInfoBytes = 8 * 2 + 4 + 4;

      uint64FeatureNum = 0;
      for (int j = 0; j < meta.getEdge_uint64_feature_num(); ++j) {
        uint64FeatureNum += edge.getUint64_feature().get(j) == null ? 0 :
                edge.getUint64_feature().get(j).size();
      }
      edgeInfoBytes += 4 + meta.getEdge_uint64_feature_num() * 4 + uint64FeatureNum * 8;

      floatFeatureNum = 0;
      for (int j = 0; j < meta.getEdge_float_feature_num(); ++j) {
        floatFeatureNum += edge.getFloat_feature().get(j) == null ? 0 :
                edge.getFloat_feature().get(j).size();
      }
      edgeInfoBytes += 4 + meta.getEdge_float_feature_num() * 4 + floatFeatureNum * 4;

      binaryFeatureNum = 0;
      for (int j = 0; j < meta.getEdge_binary_feature_num(); ++j) {
        binaryFeatureNum += edge.getBinary_feature().get(j) == null ? 0 :
                edge.getBinary_feature().get(j).length();
      }
      edgeInfoBytes += 4 + meta.getEdge_binary_feature_num() * 4 + binaryFeatureNum;

      edgesInfoBytes.add(edgeInfoBytes);
      totalEdgeInfoBytes += edgeInfoBytes;
    }

    // get blockBytes
    blockBytes = 4 + (4 + 4 * edgesInfoBytes.size()) + nodeInfoBytes + totalEdgeInfoBytes;

    // output
    writer.writeInt(blockBytes);
    writer.writeInt(nodeInfoBytes);
    writer.writeLong(block.getNode_id());
    writer.writeInt(block.getNode_type());
    writer.writeFloat(block.getNode_weight());

    // get edge group info
    writer.writeInt(meta.getEdge_type_num());
    ArrayList<Integer> edgeGroupNum = new ArrayList<>();
    ArrayList<Float> edgeGroupWeight = new ArrayList<>();
    for (int i = 0; i < meta.getEdge_type_num(); ++i) {
      edgeGroupNum.add(block.getNeighbor().get(i) == null ? 0 : block.getNeighbor().get(i).size());
      float sumWeight = 0.0f;
      if (block.getNeighbor().get(i) != null) {
        for (Long nodeId : block.getNeighbor().get(i).keySet()) {
          sumWeight += block.getNeighbor().get(i).get(nodeId);
        }
      }
      edgeGroupWeight.add(sumWeight);
    }
    writer.writeIntList(edgeGroupNum);
    writer.writeFloatList(edgeGroupWeight);

    // get neighbor info
    ArrayList<Long> neighborIdList = new ArrayList<>();
    ArrayList<Float> neighborWeightList = new ArrayList<>();
    for (int i = 0; i < meta.getEdge_type_num(); ++i) {
      if (block.getNeighbor().get(i) != null) {
        for (Long nodeId : block.getNeighbor().get(i).keySet()) {
          neighborIdList.add(nodeId);
          neighborWeightList.add(block.getNeighbor().get(i).get(nodeId));
        }
      }
    }
    writer.writeLongList(neighborIdList);
    writer.writeFloatList(neighborWeightList);

    // get features info
    writer.writeInt(meta.getNode_uint64_feature_num());
    for (int i = 0; i < meta.getNode_uint64_feature_num(); ++i) {
      writer.writeInt(block.getUint64_feature().get(i) == null ? 0 :
              block.getUint64_feature().get(i).size());
    }
    for (int i = 0; i < meta.getNode_uint64_feature_num(); ++i) {
      if (block.getUint64_feature().get(i) != null) {
        for (int j = 0; j < block.getUint64_feature().get(i).size(); ++j) {
          writer.writeLong(block.getUint64_feature().get(i).get(j));
        }
      }
    }

    writer.writeInt(meta.getNode_float_feature_num());
    for (int i = 0; i < meta.getNode_float_feature_num(); ++i) {
      writer.writeInt(block.getFloat_feature().get(i) == null ? 0 :
              block.getFloat_feature().get(i).size());
    }
    for (int i = 0; i < meta.getNode_float_feature_num(); ++i) {
      if (block.getFloat_feature().get(i) != null) {
        for (int j = 0; j < block.getFloat_feature().get(i).size(); ++j) {
          writer.writeFloat(block.getFloat_feature().get(i).get(j));
        }
      }
    }

    writer.writeInt(meta.getNode_binary_feature_num());
    for (int i = 0; i < meta.getNode_binary_feature_num(); ++i) {
      writer.writeInt(block.getBinary_feature().get(i) == null ? 0 :
              block.getBinary_feature().get(i).length());
    }
    for (int i = 0; i < meta.getNode_binary_feature_num(); ++i) {
      if (block.getBinary_feature().get(i) != null) {
        writer.writeString(block.getBinary_feature().get(i));
      }
    }

    // edge info
    writer.writeInt(block.getEdge().size());
    writer.writeIntList(edgesInfoBytes);
    for (Edge edge : block.getEdge()) {
      writer.writeLong(edge.getSrc_id());
      writer.writeLong(edge.getDst_id());
      writer.writeInt(edge.getEdge_type());
      writer.writeFloat(edge.getWeight());
      // edge feature info
      writer.writeInt(meta.getEdge_uint64_feature_num());
      for (int i = 0; i < meta.getEdge_uint64_feature_num(); ++i) {
        writer.writeInt(edge.getUint64_feature().get(i) == null ? 0 :
                edge.getUint64_feature().get(i).size());
      }
      for (int i = 0; i < meta.getEdge_uint64_feature_num(); ++i) {
        if (edge.getUint64_feature().get(i) != null) {
          for (int j = 0; j < edge.getUint64_feature().get(i).size(); ++j) {
            writer.writeLong(edge.getUint64_feature().get(i).get(j));
          }
        }
      }

      writer.writeInt(meta.getEdge_float_feature_num());
      for (int i = 0; i < meta.getEdge_float_feature_num(); ++i) {
        writer.writeInt(edge.getFloat_feature().get(i) == null ? 0 :
                edge.getFloat_feature().get(i).size());
      }
      for (int i = 0; i < meta.getEdge_float_feature_num(); ++i) {
        if (edge.getFloat_feature().get(i) != null) {
          for (int j = 0; j < edge.getFloat_feature().get(i).size(); ++j) {
            writer.writeFloat(edge.getFloat_feature().get(i).get(j));
          }
        }
      }

      writer.writeInt(meta.getEdge_binary_feature_num());
      for (int i = 0; i < meta.getEdge_binary_feature_num(); ++i) {
        writer.writeInt(edge.getBinary_feature().get(i) == null ? 0 :
                edge.getBinary_feature().get(i).length());
      }
      for (int i = 0; i < meta.getEdge_binary_feature_num(); ++i) {
        if (edge.getBinary_feature().get(i) != null) {
          writer.writeString(edge.getBinary_feature().get(i));
        }
      }
    }
    return writer.getBytes();
  }
}
