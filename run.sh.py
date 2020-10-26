#CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 6 -g 1 -b 1024 --graph_emb_dim 256 --graph_encoder_num_layers_gnn 6 --sinkhorn_temp 10 # target 0.1
#CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 7 -g 1 -b 1024 --graph_emb_dim 256 --graph_encoder_num_layers_gnn 6 --sinkhorn_temp 10 # target 0.01
#CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 8 -g 1 -b 1024 --graph_emb_dim 256 --graph_encoder_num_layers_gnn 6 --sinkhorn_temp 10 # target 0.08

CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 37 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.001 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --postprocess_method 1 --postprocess_temp 5
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 39 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.01 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 #old perm loss
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 40 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.01 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --nolin relu #old perm loss
CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 41 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.001 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --postprocess_method 1 --postprocess_temp 10 # old
# 42 like 41 but with new entropy loss (added row, col sum loss)
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 43 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.03 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 # new entropy loss
CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 44 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.05 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024
CUDA_VISIBLE_DEVICES=10 python graphae/main.py -i 45 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 32 --batch_norm --alpha 0.05 --progress_bar --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 46 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 32 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 47 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 32 --batch_norm --alpha 0.1 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 48 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.05 --lr 0.0004 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 49 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.1 --lr 0.0004 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --meta_node_dim 64 --node_emb_dim 256 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 55 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.05 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 &
CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 56 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.05 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &
CUDA_VISIBLE_DEVICES=10 python graphae/main.py -i 57 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.1 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 &
CUDA_VISIBLE_DEVICES=11 python graphae/main.py -i 58 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 59 -g 1 -b 1024 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 60 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 --nonlin relu &
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 61 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 9 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=5 python graphae/main.py -i 62 -g 1 -b 1024 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 1024 --graph_encoder_hidden_dim_fnn 2048 &
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 63 -g 1 -b 1024 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 6 --permuter_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 1024 --graph_encoder_hidden_dim_fnn 2048 &
# fixed metric weights for 16 nodes
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 64 -g 1 -b 1024 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 9 --node_dim 1024  --graph_encoder_num_layers_fnn 6 &

CUDA_VISIBLE_DEVICES=0 python graphae/main.py -i 65 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 32 --batch_norm --alpha 0.075 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 &

# no scaling for 1 constrain
CUDA_VISIBLE_DEVICES=7 python graphae/main.py -i 67 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.1 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 &
CUDA_VISIBLE_DEVICES=8 python graphae/main.py -i 68 -g 1 -b 1024 --graph_emb_dim 1024 --max_num_nodes 32 --batch_norm --alpha 0.1 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 9 --node_dim 512 &
CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 69 -g 1 -b 1024 --graph_emb_dim 512 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 --graph_encoder_hidden_dim_fnn 1024 &

# 5x scaling for 1 constrain

CUDA_VISIBLE_DEVICES=12 python graphae/main.py -i 70 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.1 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &
CUDA_VISIBLE_DEVICES=13 python graphae/main.py -i 71 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.1 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 9 --node_dim 512 &
CUDA_VISIBLE_DEVICES=10 python graphae/main.py -i 72 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.5 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &
# with perm npomralization again
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 73 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.5 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &
CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 74 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 75 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 9 --node_dim 512 &
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 76 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 256 &

#alpha decay

CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 77 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.001 --lr 0.0002 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &
CUDA_VISIBLE_DEVICES=7 python graphae/main.py -i 78 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.0025 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --graph_encoder_num_layers_gnn 9 --node_dim 1024 --graph_encoder_num_layers_fnn 6 --edge_predictor_hidden_dim 4096 &

# no alpha decay. bigger edge decoder
CUDA_VISIBLE_DEVICES=13 python graphae/main.py -i 79 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --graph_encoder_num_layers_gnn 9 --node_dim 1024 --graph_encoder_num_layers_fnn 6 --edge_predictor_hidden_dim 4096 &
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 81 -g 1 -b 64 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --graph_encoder_num_layers_gnn 9 --node_dim 1024 --graph_encoder_num_layers_fnn 6 --edge_predictor_hidden_dim 4096 &

CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 83 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --graph_encoder_num_layers_gnn 9 --node_dim 1024 --graph_encoder_num_layers_fnn 6 --edge_predictor_hidden_dim 4096 &
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 84 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 512 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --edge_predictor_hidden_dim 1024 --progress_bar
# encoder batchnorm
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 85 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 512 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --edge_predictor_hidden_dim 1024 --progress_bar
# weight sharing encoder gnn
CUDA_VISIBLE_DEVICES=5 python graphae/main.py -i 86 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 512 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --edge_predictor_hidden_dim 1024 &
# two more layers in GIN mlp + more channels
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 87 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --edge_predictor_hidden_dim 1024 &

# removed weight sharing kept batchnorm
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 88 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.4 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --graph_encoder_num_layers_gnn 9 --node_dim 1024 --graph_encoder_num_layers_fnn 6 --edge_predictor_hidden_dim 4096 &
CUDA_VISIBLE_DEVICES=7 python graphae/main.py -i 89 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.4 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 512 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --edge_predictor_hidden_dim 1024 &
CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 83 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 2048 --graph_encoder_hidden_dim_fnn 2048 --meta_node_decoder_hidden_dim 2048 --graph_encoder_num_layers_gnn 9 --node_dim 1024 --graph_encoder_num_layers_fnn 6 --edge_predictor_hidden_dim 4096 &
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 90 -g 1 -b 512 --graph_emb_dim 1024 --max_num_nodes 16 --batch_norm --alpha 0.25 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 1024 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 2048 --edge_predictor_num_layers 5 --graph_encoder_num_layers_gnn 7 --node_dim 512 &  #like 59

CUDA_VISIBLE_DEVICES=8 python graphae/main.py -i 91 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --graph_encoder_hidden_dim_gnn 512 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --progress_bar


CUDA_VISIBLE_DEVICES=10 python graphae/main.py -i 93 -g 1 -b 512 --graph_emb_dim 256 --max_num_nodes 16 --batch_norm --alpha 0.2 --lr 0.0001 --permuter_num_layers 5 --permuter_hidden_dim 1024 --edge_predictor_num_layers 5 --graph_encoder_hidden_dim_gnn 512 --graph_encoder_hidden_dim_fnn 1024 --meta_node_decoder_hidden_dim 1024 --graph_encoder_num_layers_gnn 7 --node_dim 256 --graph_encoder_num_layers_fnn 3 --edge_predictor_hidden_dim 1024 --progress_bar

#######  saves 7

CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 1 --batch_norm --progress_bar
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 2 --batch_norm --progress_bar --graph_emb_dim 1024 --alpha 0.01
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 3 --batch_norm --progress_bar --lr 0.00005 -b 256

# new matching loss
CUDA_VISIBLE_DEVICES=5 python graphae/main.py -i 4--batch_norm --progress_bar --lr 0.00005 -b 256
CUDA_VISIBLE_DEVICES=5 python graphae/main.py -i 4--batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.01

# random smiles fix

CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 5 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.0

CUDA_VISIBLE_DEVICES=7 python graphae/main.py -i 6 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.01
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 7 --batch_norm --lr 0.00005 -b 256 --alpha 0.01 --graph_emb_dim 1024
CUDA_VISIBLE_DEVICES=8 python graphae/main.py -i 8 --batch_norm --lr 0.0002 -b 256 --alpha 0.01 --graph_emb_dim 1024 &
CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 9 --batch_norm --lr 0.0002 -b 256 --alpha 0.01 --graph_emb_dim 1024 &
CUDA_VISIBLE_DEVICES=10 python graphae/main.py -i 10 --batch_norm --lr 0.0002 -b 256 --alpha 0.1 --graph_emb_dim 1024 &
CUDA_VISIBLE_DEVICES=11 python graphae/main.py -i 11 --batch_norm --lr 0.0002 -b 256 --alpha 1.0 --graph_emb_dim 1024 &
CUDA_VISIBLE_DEVICES=12 python graphae/main.py -i 12 --max_num_nodes 32 --batch_norm --lr 0.0002 -b 256 --alpha 1.0 --graph_emb_dim 1024 &

CUDA_VISIBLE_DEVICES=11 python graphae/main.py -i 13 --batch_norm --lr 0.0002 -b 256 --alpha 0.1 --graph_emb_dim 512 &

CUDA_VISIBLE_DEVICES=5 python graphae/main.py -i 14 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.1 &
CUDA_VISIBLE_DEVICES=0 python graphae/main.py -i 15 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.1  --graph_emb_dim 1024 &
CUDA_VISIBLE_DEVICES=1 python graphae/main.py -i 16 --max_num_nodes 32 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.1 &
CUDA_VISIBLE_DEVICES=2 python graphae/main.py -i 17 --max_num_nodes 32 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.1 --graph_emb_dim 1024 &
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 18 --batch_norm --progress_bar --lr 0.00005 -b 256 --alpha 0.1  --graph_emb_dim 1024 --node_dim 512 &

CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 19 --batch_norm --lr 0.00005 -b 256 --alpha 0.0 &
# tf decay on 0.95 adj_acc
CUDA_VISIBLE_DEVICES=7 python graphae/main.py -i 20 --batch_norm --lr 0.00005 -b 256 --alpha 0.0 &

# only one layer rnn out
CUDA_VISIBLE_DEVICES=6 python graphae/main.py -i 27 --batch_norm --lr 0.00005 -b 256 --alpha 0.0 &
CUDA_VISIBLE_DEVICES=0 python graphae/main.py -i 35 --batch_norm --lr 0.00005 -b 256 --alpha 0.1 &
CUDA_VISIBLE_DEVICES=3 python graphae/main.py -i 36 --batch_norm --lr 0.00005 -b 256 --alpha 0.0 --max_num_nodes 32 &
CUDA_VISIBLE_DEVICES=4 python graphae/main.py -i 37 --batch_norm --lr 0.00005 -b 256 --alpha 0.1 --max_num_nodes 32 &
CUDA_VISIBLE_DEVICES=5 python graphae/main.py -i 38 --batch_norm --lr 0.00005 -b 256 --alpha 0.1 --max_num_nodes 32 --graph_emb_dim 1024 --node_dim 512 &

CUDA_VISIBLE_DEVICES=7 python graphae/main.py -i 39 --batch_norm --lr 0.00005 -b 256 --alpha 1.0 --progress_bar
# no detach of enocded node embs
CUDA_VISIBLE_DEVICES=8 python graphae/main.py -i 40 --batch_norm --lr 0.00005 -b 256 --alpha 1.0 --progress_bar
# noise on enocded node embs
CUDA_VISIBLE_DEVICES=9 python graphae/main.py -i 41 --batch_norm --lr 0.00005 -b 256 --alpha 1.0 --progress_bar
# multinominal
CUDA_VISIBLE_DEVICES=12 python graphae/main.py -i 42 --batch_norm --lr 0.00005 -b 256 --alpha 1.0 --progress_bar

