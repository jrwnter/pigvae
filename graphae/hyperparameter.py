import os
DEFAULT_DATA_PATH = "/home/ggwaq/projects/graph_vae/smiles_with_features.csv.gz"
DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "saves13")


def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # GENERAL
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=5000, type=int)
    parser.add_argument("--num_eval_samples", default=8192*8, type=int)
    parser.add_argument("--num_samples_per_epoch", default=800000000, type=int)
    parser.add_argument("--num_samples_per_epoch_inc", default=2000000, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true')
    parser.set_defaults(test=False)
    parser.set_defaults(progress_bar=False)

    # TRAINING
    parser.add_argument("--resume_ckpt", default="", type=str)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--lr_scheduler_factor", default=0.5, type=float)
    parser.add_argument("--lr_scheduler_patience", default=2, type=int)
    parser.add_argument("--lr_scheduler_cooldown", default=5, type=int)
    parser.add_argument("--start_tf", default=0.9, type=float)
    parser.add_argument("--tf_decay_factor", default=0.9, type=float)
    parser.add_argument("--tf_decay_freq", default=20, type=int)
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--alpha", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--emb_noise", default=0.05, type=float)
    parser.add_argument("--vae", dest='vae', action='store_true')
    parser.set_defaults(vae=False)

    # GENERAL GRAPH PROPERTIES
    parser.add_argument("--max_num_nodes", default=32, type=int)
    parser.add_argument("--num_node_features", default=20, type=int)
    parser.add_argument("--num_edge_features", default=4, type=int)
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true')
    parser.set_defaults(batch_norm=False)

    parser.add_argument("--nonlin", default="relu", type=str)

    # GRAPH ENCODER
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--graph_encoder_hidden_dim", default=512, type=int)
    parser.add_argument("--graph_encoder_k_dim", default=64, type=int)
    parser.add_argument("--graph_encoder_v_dim", default=64, type=int)
    parser.add_argument("--graph_encoder_num_heads", default=32, type=int)
    parser.add_argument("--graph_encoder_ppf_hidden_dim", default=1024, type=int)
    parser.add_argument("--graph_encoder_num_layers", default=16, type=int)

    # GRAPH DECODER

    parser.add_argument("--graph_decoder_hidden_dim", default=512, type=int)
    parser.add_argument("--graph_decoder_k_dim", default=64, type=int)
    parser.add_argument("--graph_decoder_v_dim", default=64, type=int)
    parser.add_argument("--graph_decoder_num_heads", default=32, type=int)
    parser.add_argument("--graph_decoder_ppf_hidden_dim", default=1024, type=int)
    parser.add_argument("--graph_decoder_num_layers", default=16, type=int)
    parser.add_argument("--graph_decoder_pos_emb_dim", default=64, type=int)


    # PROPERTY PREDICTOR
    parser.add_argument("--property_predictor_hidden_dim", default=1024, type=int)
    parser.add_argument("--num_properties", default=8, type=int)


    # DATA
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--num_rows", default=None, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--shuffle", default=1, type=int)

    return parser

