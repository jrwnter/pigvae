import os
DEFAULT_DATA_PATH = "/home/ggwaq/projects/graph_vae/smiles_atom_count2.csv"
DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "saves2")


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
    parser.add_argument('-e', '--num_epochs', default=50, type=int)
    parser.add_argument("--num_eval_samples", default=1024, type=int)
    parser.add_argument("--eval_freq", default=200, type=int)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true')
    parser.set_defaults(test=False)
    parser.set_defaults(progress_bar=False)

    # TRAINING
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--lr_scheduler_factor", default=0.5, type=float)
    parser.add_argument("--lr_scheduler_patience", default=2, type=int)
    parser.add_argument("--lr_scheduler_cooldown", default=5, type=int)
    parser.add_argument("--vae", dest='vae', action='store_true')
    parser.set_defaults(vae=False)

    # GENERAL GRAPH PROPERTIES
    parser.add_argument("--max_num_nodes", default=16, type=int)
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true')
    parser.set_defaults(batch_norm=False)

    # ENCODER
    parser.add_argument("--emb_dim", default=512, type=int)
    parser.add_argument("--node_dim", default=256, type=int)
    parser.add_argument("--graph_encoder_hidden_dim", default=512, type=int)
    parser.add_argument("--graph_encoder_num_layers", default=6, type=int)
    parser.add_argument("--node_aggregator_num_layers", default=3, type=int)
    parser.add_argument("--node_aggregator_hidden_dim", default=1024, type=int)

    # DECODER
    parser.add_argument("--meta_node_dim", default=256, type=int)
    parser.add_argument("--meta_node_decoder_hidden_dim", default=2048, type=int)
    parser.add_argument("--meta_node_decoder_num_layers", default=5, type=int)

    parser.add_argument("--edge_predictor_hidden_dim", default=2048, type=int)
    parser.add_argument("--edge_predictor_num_layers", default=5, type=int)

    parser.add_argument("--node_decoder_hidden_dim", default=2048, type=int)
    parser.add_argument("--node_decoder_num_layers", default=5, type=int)

    # DATA
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--num_rows", default=None, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_atom_features", default=24, type=int) #24
    parser.add_argument("--shuffle", default=1, type=int)

    return parser
