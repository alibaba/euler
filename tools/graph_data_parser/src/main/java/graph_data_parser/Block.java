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

import java.util.ArrayList;
import java.util.HashMap;

class Meta {
  private int node_type_num;
  private int edge_type_num;
  private int node_uint64_feature_num;
  private int node_float_feature_num;
  private int node_binary_feature_num;
  private int edge_uint64_feature_num;
  private int edge_float_feature_num;
  private int edge_binary_feature_num;

  public Meta() {

  }

  public int getNode_type_num() {
    return node_type_num;
  }

  public void setNode_type_num(int v) {
    node_type_num = v;
  }

  public int getEdge_type_num() {
    return edge_type_num;
  }

  public void setEdge_type_num(int v) {
    edge_type_num = v;
  }

  public int getNode_uint64_feature_num() {
    return node_uint64_feature_num;
  }

  public void setNode_uint64_feature_num(int v) {
    node_uint64_feature_num = v;
  }

  public int getNode_float_feature_num() {
    return node_float_feature_num;
  }

  public void setNode_float_feature_num(int v) {
    node_float_feature_num = v;
  }

  public int getNode_binary_feature_num() {
    return node_binary_feature_num;
  }

  public void setNode_binary_feature_num(int v) {
    node_binary_feature_num = v;
  }

  public int getEdge_uint64_feature_num() {
    return edge_uint64_feature_num;
  }

  public void setEdge_uint64_feature_num(int v) {
    edge_uint64_feature_num = v;
  }

  public int getEdge_float_feature_num() {
    return edge_float_feature_num;
  }

  public void setEdge_float_feature_num(int v) {
    edge_float_feature_num = v;
  }

  public int getEdge_binary_feature_num() {
    return edge_binary_feature_num;
  }

  public void setEdge_binary_feature_num(int v) {
    edge_binary_feature_num = v;
  }
}

class Edge {
  private long src_id;
  private long dst_id;
  private int edge_type;
  private float weight;
  private HashMap<Integer, ArrayList<Long>> uint64_feature;
  private HashMap<Integer, ArrayList<Float>> float_feature;
  private HashMap<Integer, String> binary_feature;

  public Edge() {
    uint64_feature = new HashMap<>();
    float_feature = new HashMap<>();
    binary_feature = new HashMap<>();
  }

  public long getSrc_id() {
    return src_id;
  }

  public void setSrc_id(long src_id) {
    this.src_id = src_id;
  }

  public long getDst_id() {
    return dst_id;
  }

  public void setDst_id(long dst_id) {
    this.dst_id = dst_id;
  }

  public int getEdge_type() {
    return edge_type;
  }

  public void setEdge_type(int edge_type) {
    this.edge_type = edge_type;
  }

  public float getWeight() {
    return weight;
  }

  public void setWeight(float weight) {
    this.weight = weight;
  }

  public HashMap<Integer, ArrayList<Long>> getUint64_feature() {
    return uint64_feature;
  }

  public void setUint64_feature(HashMap<Integer, ArrayList<Long>> uint64_feature) {
    this.uint64_feature = uint64_feature;
  }

  public HashMap<Integer, ArrayList<Float>> getFloat_feature() {
    return float_feature;
  }

  public void setFloat_feature(HashMap<Integer, ArrayList<Float>> float_feature) {
    this.float_feature = float_feature;
  }

  public HashMap<Integer, String> getBinary_feature() {
    return binary_feature;
  }

  public void setBinary_feature(HashMap<Integer, String> binary_feature) {
    this.binary_feature = binary_feature;
  }
}

public class Block {
  private long node_id;
  private int node_type;
  private float node_weight;
  private HashMap<Integer, HashMap<Long, Float>> neighbor;
  private HashMap<Integer, ArrayList<Long>> uint64_feature;
  private HashMap<Integer, ArrayList<Float>> float_feature;
  private HashMap<Integer, String> binary_feature;
  private ArrayList<Edge> edge;

  public Block() {
    neighbor = new HashMap<>();
    uint64_feature = new HashMap<>();
    float_feature = new HashMap<>();
    binary_feature = new HashMap<>();
    edge = new ArrayList<>();
  }

  public long getNode_id() {
    return node_id;
  }

  public void setNode_id(long node_id) {
    this.node_id = node_id;
  }

  public int getNode_type() {
    return node_type;
  }

  public void setNodeType(int node_type) {
    this.node_type = node_type;
  }

  public float getNode_weight() {
    return node_weight;
  }

  public void setNode_weight(float node_weight) {
    this.node_weight = node_weight;
  }

  public HashMap<Integer, HashMap<Long, Float>> getNeighbor() {
    return neighbor;
  }

  public void setNeighbor(HashMap<Integer, HashMap<Long, Float>> neighbor) {
    this.neighbor = neighbor;
  }

  public HashMap<Integer, ArrayList<Long>> getUint64_feature() {
    return uint64_feature;
  }

  public void setUint64_feature(HashMap<Integer, ArrayList<Long>> uint64_feature) {
    this.uint64_feature = uint64_feature;
  }

  public HashMap<Integer, ArrayList<Float>> getFloat_feature() {
    return float_feature;
  }

  public void setFloat_feature(HashMap<Integer, ArrayList<Float>> float_feature) {
    this.float_feature = float_feature;
  }

  public HashMap<Integer, String> getBinary_feature() {
    return binary_feature;
  }

  public void setBinary_feature(HashMap<Integer, String> binary_feature) {
    this.binary_feature = binary_feature;
  }

  public ArrayList<Edge> getEdge() {
    return edge;
  }

  public void setEdge(ArrayList<Edge> edge) {
    this.edge = edge;
  }
}
