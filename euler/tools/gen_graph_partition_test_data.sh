/bin/bash gen_paritioned_data.sh ../../tools/test_data/graph.json ../../tools/test_data/meta /tmp/gp_euler 1
cp /tmp/gp_euler/Edge/0_0.dat /tmp/gp_euler/Edge/0_1.dat
cp /tmp/gp_euler/Node/0_0.dat /tmp/gp_euler/Node/0_1.dat
cp /tmp/gp_euler/Index/att/0_0.dat /tmp/gp_euler/Index/att/0_1.dat
cp /tmp/gp_euler/Index/edge_att/0_0.dat /tmp/gp_euler/Index/edge_att/0_1.dat
cp /tmp/gp_euler/Index/edge_type/0_0.dat /tmp/gp_euler/Index/edge_type/0_1.dat
cp /tmp/gp_euler/Index/edge_value/0_0.dat /tmp/gp_euler/Index/edge_value/0_1.dat
cp /tmp/gp_euler/Index/graph_label/0_0.dat /tmp/gp_euler/Index/graph_label/0_1.dat
cp /tmp/gp_euler/Index/node_type/0_0.dat /tmp/gp_euler/Index/node_type/0_1.dat
cp /tmp/gp_euler/Index/price/0_0.dat /tmp/gp_euler/Index/price/0_1.dat

