import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process hyper-parameters')

    parser.add_argument('--sweep', type=bool, default=False, help='choose whether to sweep or not')
    parser.add_argument('--pretraining', type=bool, default=True, help='load pretrained checkpoint for main training')
    parser.add_argument('--run_name', type=str, default="direc", help='name of the run')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='early stopping patience')
    parser.add_argument('--do_validation', type=bool, default=True, help="Do validation or not")
    parser.add_argument('--seed', type=int, default=2026, help='random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--data_path', type=str, default="./dataset/", help='path to dataset')
    parser.add_argument('--dataset', type=str, default='movie_to_music', help='Choose a dataset from [movie_to_music, book_to_movie, book_to_music]')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--noise_steps', type=int, default=1500, help='noise steps')
    parser.add_argument('--cold_start_ratio', type=float, default=0.2, help='cold start user ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='validation ratio among training set')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--lamda', type=float, default=0.7, help='weight for diffusion loss')
    parser.add_argument('--history_len', type=int, default=20, help='Max history length for source domain user interaction')    
    parser.add_argument('--ddim_steps', type=int, default=100, help='Number of steps for DDIM')

    # print arguments
    attribute_dict = dict(vars(parser.parse_args()))
    print('*' * 32 + ' Experiment setting ' + '*' * 32)
    for k, v in attribute_dict.items():
        print(k + ' : ' + str(v))
    print('*' * 32 + ' Experiment setting ' + '*' * 32)

    return parser.parse_args()