import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)

    parser.add_argument("--num_epoch", type=int, default=4)`
    parser.add_argument("--train_triplets_per_epoch", type=int, default=100000)
    parser.add_argument("--eval_triplets_per_epoch", type=int, default=10000)
    parser.add_argument("--ratio_hard_neg", type=float, default=0.4)
    parser.add_argument("--ratio_nn_neg", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    return args
