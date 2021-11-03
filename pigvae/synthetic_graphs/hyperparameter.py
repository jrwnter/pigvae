import os


DEFAULT_SAVE_DIR = os.path.join(os.getcwd())


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
    parser.add_argument("--num_eval_samples", default=8192, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true')
    parser.set_defaults(test=False)
    parser.set_defaults(progress_bar=False)

    # TRAINING
    parser.add_argument("--resume_ckpt", default="", type=str)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.00005, type=float)
    parser.add_argument("--kld_loss_scale", default=0.001, type=float)
    parser.add_argument("--perm_loss_scale", default=0.5, type=float)
    parser.add_argument("--property_loss_scale", default=0.1, type=float)
    parser.add_argument("--vae", dest='vae', action='store_true')
    parser.set_defaults(vae=False)

    # GENERAL GRAPH PROPERTIES
    parser.add_argument("--num_node_features", default=1, type=int)
    parser.add_argument("--num_edge_features", default=6, type=int)

    # GRAPH ENCODER
    parser.add_argument("--emb_dim", default=64, type=int)
    parser.add_argument("--graph_encoder_hidden_dim", default=256, type=int)
    parser.add_argument("--graph_encoder_k_dim", default=64, type=int)
    parser.add_argument("--graph_encoder_v_dim", default=64, type=int)
    parser.add_argument("--graph_encoder_num_heads", default=16, type=int)
    parser.add_argument("--graph_encoder_ppf_hidden_dim", default=1024, type=int)
    parser.add_argument("--graph_encoder_num_layers", default=16, type=int)

    # GRAPH DECODER

    parser.add_argument("--graph_decoder_hidden_dim", default=256, type=int)
    parser.add_argument("--graph_decoder_k_dim", default=64, type=int)
    parser.add_argument("--graph_decoder_v_dim", default=64, type=int)
    parser.add_argument("--graph_decoder_num_heads", default=16, type=int)
    parser.add_argument("--graph_decoder_ppf_hidden_dim", default=1024, type=int)
    parser.add_argument("--graph_decoder_num_layers", default=16, type=int)
    parser.add_argument("--graph_decoder_pos_emb_dim", default=64, type=int)


    # PROPERTY PREDICTOR
    parser.add_argument("--property_predictor_hidden_dim", default=256, type=int)
    parser.add_argument("--num_properties", default=1, type=int)


    # DATA
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--shuffle", default=1, type=int)
    parser.add_argument("--graph_family", default="barabasi_albert", type=str)
    parser.add_argument("--n_min", default=12, type=int)
    parser.add_argument("--n_max", default=20, type=int)
    parser.add_argument("--p_min", default=0.4, type=float)
    parser.add_argument("--p_max", default=0.6, type=float)
    parser.add_argument("--m_min", default=1, type=int)
    parser.add_argument("--m_max", default=5, type=int)

    return parser

