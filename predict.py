from utils import prepare_input, nums_to_char

import torch
import numpy as np


def predict_dan_kyu(model, csv_path):
    # Load the corresponding dataset
    df = open(csv_path).read().splitlines()
    games_id = [i.split(',',2)[0] for i in df]
    games = [i.split(',',2)[-1] for i in df]

    x_testing = []

    for game in games:
        moves_list = game.split(',')
        x_testing.append(prepare_input(moves_list))

    x_testing = torch.Tensor(x_testing)
    pred = model(x_testing)
    _, maxk = torch.topk(pred, 5, dim = -1)
    prediction_chars = nums_to_char(maxk.detach().numpy())

    # Save results to public_submission.csv
    with open('./public_submission.csv','a') as f:
        for index in range(len(prediction_chars)):
            answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
            f.write(answer_row)

def predict_play_style(model, csv_path):
        # Load the corresponding dataset
    df = open(csv_path).read().splitlines()
    games_id = [i.split(',',1)[0] for i in df]
    games = [i.split(',',1)[-1] for i in df]

    x_testing = []

    for game in games:
        moves_list = game.split(',')
        x_testing.append(prepare_input(moves_list))

    x_testing = torch.Tensor(x_testing)
    pred = model(x_testing)
    _, maxk = torch.topk(pred, 1, dim = -1)
    maxk = maxk.detach().numpy().reshape(-1)

    # Save results to public_submission.csv
    with open('./public_submission.csv','a') as f:
        for index, number in enumerate(maxk):
            answer_row = games_id[index] + ',' + str(number+1) + '\n'
            f.write(answer_row)

checkpoint = torch.load('./result/exp/playstyle_CNN_epoch1.pth')
model = checkpoint['model']
model.load_state_dict(checkpoint['model_weights'])
print(model)
predict_play_style(model, './data/Test/play_style_test_public.csv')