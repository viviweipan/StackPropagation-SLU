from utils.module import ModelManager
from utils.loader import DatasetManager
from utils.process import Processor

import torch

import os
import json
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()

# Training parameters.
parser.add_argument('--data_dir', '-dd', type=str, default='data/atis')
parser.add_argument('--save_dir', '-sd', type=str, default='save_0827')
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=300)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

# model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = DatasetManager(args)
    dataset.quick_build()
    dataset.show_summary()


    result_per_intent = Processor.validate_per_intent(
        os.path.join(args.save_dir, "model/model.pkl"),
        os.path.join(args.save_dir, "model/dataset.pkl"),
        args.batch_size)

    print(f'Result in directory {args.save_dir}')
    print('*'*50)
    print('intent slot_f1 intent_acc sen_acc num_utt')
    print('overall' + ' ' + str(result_per_intent['overall'][0])+ ' ' + str(result_per_intent['overall'][1])
          + ' ' + str(result_per_intent['overall'][2])
          + ' ' + str(result_per_intent['overall'][3]))

    for intent in result_per_intent:
        if intent != 'overall':
            print(str(intent) + ' ' + str(result_per_intent[intent][0])+ ' ' + str(result_per_intent[intent][1])+ ' ' + str(result_per_intent[intent][2])
              + ' ' + str(result_per_intent[intent][3]))