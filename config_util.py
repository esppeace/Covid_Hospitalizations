from argparse import ArgumentParser

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument("--state_data_dir",  type=str, required=True)
    parser.add_argument("--county_data_dir",  type=str, required=True)
    parser.add_argument('--state_dir', default="states_vaccine")
    parser.add_argument('--counties_dir', default="counties_vaccine")

    parser.add_argument('--model', default='transformer')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--use_economy', action='store_true', default=False)
    parser.add_argument('--use_deaths', action='store_true', default=False)
    parser.add_argument('--use_cases', action='store_true', default=False)
    parser.add_argument('--use_hospitalization', action='store_true', default=False)
    parser.add_argument('--use_hospitalization_smoothing', action='store_true', default=False)
    parser.add_argument('--hospitalization_list', nargs='+', default=['hospitalization_total', 'hospitalization_total_adult', 'hospitalization_total_pediatric'])
    parser.add_argument('--hospitalization_prediction_list', nargs='+', default=['hospitalization_total', 'hospitalization_total_adult', 'hospitalization_total_pediatric'])
    parser.add_argument('--use_how_many_days', type=int, default=7)
    parser.add_argument('--num_of_week', type=int, default=4)
    parser.add_argument('--remove_days', type=int, default=45)


    parser.add_argument('--standardize', action='store_true', default=False)

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--not_using_states', action='store_true', default=False)
    parser.add_argument('--not_using_counties', action='store_true', default=False)
    
    parser.add_argument('--use_month', action='store_true', default=False)
    parser.add_argument('--split_train_val', action='store_true', default=False)

    parser.add_argument('--loss', type = str, default='mse')
    parser.add_argument('--use_smoothing_for_train', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_models', type=int, default=5)

    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--logging_interval', type=int, default=-1)
    parser.add_argument('--state_remove_day', type=int, default=45)
    parser.add_argument('--county_remove_day', type=int, default=45)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--train_val_size', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--model-dir', type = str, default='./lstm_checkpoint',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--lambda1', type=float, default=0.5)
    
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])

    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=7)
    parser.add_argument('--num_encoder_layer', type=int, default=2)
    parser.add_argument('--feedforward_dim', type=int, default=2048)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--no_norm', action='store_false', default=True)
    parser.add_argument('--half_epoch', type=int, default=50)
    parser.add_argument('--huber_beta', type=float, default=1.0)



    args = parser.parse_args()

    
    print(args)

    return args
