import os
DEFAULT_DATA_PATH = "/home/ggwaq/projects/graph_vae/smiles_with_features.csv.gz"
DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "saves12")


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
    parser.add_argument("--num_eval_samples", default=50000, type=int)
    parser.add_argument("--num_samples_per_epoch", default=4000000, type=int)
    parser.add_argument("--num_samples_per_epoch_inc", default=2000000, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true')
    parser.set_defaults(test=False)
    parser.set_defaults(progress_bar=False)

    # TRAINING
    parser.add_argument("--resume_ckpt", default="", type=str)
    parser.add_argument("-b", "--batch_size", default=256, type=int)
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
    parser.add_argument("--max_num_nodes", default=16, type=int)
    parser.add_argument("--num_node_features", default=20, type=int)
    parser.add_argument("--num_edge_features", default=4, type=int)
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true')
    parser.set_defaults(batch_norm=False)

    parser.add_argument("--nonlin", default="lrelu", type=str)

    # GRAPH ENCODER
    parser.add_argument("--node_dim", default=256, type=int)
    parser.add_argument("--graph_encoder_hidden_dim", default=1024, type=int)
    parser.add_argument("--graph_encoder_num_layers", default=4, type=int)

    # GRAPH DECODER

    parser.add_argument("--node_emb_decoder_hidden_dim", default=1024, type=int)
    parser.add_argument("--node_emb_decoder_num_layers", default=4, type=int)

    parser.add_argument("--edge_decoder_hidden_dim", default=1024, type=int)
    parser.add_argument("--edge_decoder_num_layers", default=3, type=int)

    parser.add_argument("--node_decoder_hidden_dim", default=1024, type=int)
    parser.add_argument("--node_decoder_num_layers", default=3, type=int)

    parser.add_argument("--postprocess_method", default=0, type=int)
    parser.add_argument("--postprocess_temp", default=1.0, type=float)

    # PROPERTY PREDICTOR
    parser.add_argument("--property_predictor_hidden_dim", default=1024, type=int)
    parser.add_argument("--property_predictor_num_layers", default=3, type=int)
    parser.add_argument("--num_properties", default=8, type=int)

    # PI ENCODER
    parser.add_argument("--emb_dim", default=1024, type=int)
    parser.add_argument("--element_emb_dim", default=32, type=int)
    parser.add_argument("--pi_encoder_hidden_dim", default=2048, type=int)
    parser.add_argument("--pi_encoder_num_layers", default=3, type=int)
    parser.add_argument("--pi_encoder_p_steps", default=5, type=int)

    # PI DECODER
    parser.add_argument("--pi_decoder_hidden_dim_fnn", default=1024, type=int)
    parser.add_argument("--pi_decoder_hidden_dim_rnn", default=1024, type=int)
    parser.add_argument("--pi_decoder_num_layers_fnn", default=3, type=int)
    parser.add_argument("--pi_decoder_num_layers_rnn", default=3, type=int)
    parser.add_argument("--pi_decoder_gumbel_tau", default=1.0, type=float)

    # PI PERMUTER
    parser.add_argument("--pi_permuter_hidden_dim", default=1024, type=int)
    parser.add_argument("--pi_permuter_num_layers", default=3, type=int)



    # DATA
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--num_rows", default=None, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--shuffle", default=1, type=int)

    return parser

