import numpy as np

chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}
chartonumbers = {k:v for k,v in enumerate(chars)}

def prepare_input(moves):
        x = np.zeros((4,19,19))
        for move in moves:
            color = move[0]
            column = coordinates[move[2]]
            row = coordinates[move[3]]
            if color == 'B':
                x[0,row,column] = 1
                x[2,row,column] = 1
            if color == 'W':
                x[1,row,column] = 1
                x[2,row,column] = 1
        if moves:
            last_move_column = coordinates[moves[-1][2]]
            last_move_row = coordinates[moves[-1][3]]
            x[3, row,column] = 1
        x[2,:,:] = np.where(x[2,:,:] == 0, 1, 0)
        return x

def prepare_label(move):
    column = coordinates[move[2]]
    row = coordinates[move[3]]
    return column*19+row

def number_to_char(number):
    print(number)
    number_1, number_2 = divmod(number, 19)
    return chartonumbers[number_1] + chartonumbers[number_2]

def nums_to_char(maxk):
    resulting_preds_chars = np.vectorize(number_to_char)(maxk)
    return resulting_preds_chars