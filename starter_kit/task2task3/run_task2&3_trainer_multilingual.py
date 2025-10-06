import argparse
import math
import os
import json
import Utils
import torch
import random
from DimABSAModel import DimABSA
from torch.nn import functional as F
from transformers import BertTokenizer, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

from Utils import create_directory, ReviewDataset, generate_batches, InferenceReviewDataset, combine_lists, replace_using_dict
from DataProcess import dataset_process, dataset_inference_process

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

restaurant_entity_labels = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
restaurant_attribute_labels= ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
restaurant_category_dict, restaurant_category_list = combine_lists(restaurant_entity_labels, restaurant_attribute_labels)

laptop_entity_labels = ['LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING', 'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY', 'GRAPHICS', 'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY', 'SHIPPING', 'SUPPORT', 'COMPANY']+ ['OUT_OF_SCOPE']
laptop_attribute_labels = ['GENERAL', 'PRICE', 'QUALITY', 'DESIGN_FEATURES', 'OPERATION_PERFORMANCE', 'USABILITY', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS']
laptop_category_dict, laptop_category_list = combine_lists(laptop_entity_labels, laptop_attribute_labels)


hotel_entity_labels = ['HOTEL', 'ROOMS', 'FACILITIES', 'ROOM_AMENITIES', 'SERVICE', 'LOCATION', 'FOOD_DRINKS']
hotel_attribute_labels = ['GENERAL', 'PRICE', 'COMFORT', 'CLEANLINESS', 'QUALITY', 'DESIGN_FEATURES', 'STYLE_OPTIONS', 'MISCELLANEOUS']
hotel_category_dict, hotel_category_list = combine_lists(hotel_entity_labels, hotel_attribute_labels)

finance_entity_labels = ['MARKET', 'COMPANY', 'BUSINESS', 'PRODUCT']
finance_attribute_labels = ['GENERAL', 'SALES', 'PROFIT', 'AMOUNT', 'PRICE', 'COST']
finance_category_dict, finance_category_list = combine_lists(finance_entity_labels, finance_attribute_labels)


# # restaurant datasets in english
# res_en_train_data_path = f"./data/eng_restaurant_train_alltasks.jsonl"
# res_en_test_data_path = f"./data/eng_restaurant_dev_task2.jsonl"
#
# # laptop datasets
# lap_en_train_data_path = f"./data/eng_laptop_train_alltasks.jsonl"
# lap_en_test_data_path = f"./data/eng_laptop_dev_task2.jsonl"
#
# # laptop datasets
# lap_zho_train_data_path = f"./data/zho_laptop_train_alltasks.jsonl"
# lap_zho_test_data_path = f"./data/zho_laptop_dev_task2.jsonl"

# dataset_path_map = {
#     'res_eng': (res_en_train_data_path, None, res_en_test_data_path),
#     'lap_eng': (lap_en_train_data_path, None, lap_en_test_data_path),
#     'lap_zho': (lap_zho_train_data_path, None, lap_zho_test_data_path),
# }

category_map = {
    'res': (restaurant_category_dict, restaurant_category_list),
    'lap': (laptop_category_dict, laptop_category_list),
    'hot': (hotel_category_dict, hotel_category_list),
    'fin': (finance_category_dict, finance_category_list),
}

out_put_file_name_map = {
    'res_eng': "pred_eng_restaurant.jsonl",
    'lap_eng': "pred_eng_laptop.jsonl",
    'lap_zho': "pred_zho_laptop.jsonl",
}

lap_filter_from_category = ["HARD_DISC", "PRICES","OPERATION&PERFORMANCE","FANS&COOLING", "FANS & COOLING", "FANS_&_COOLING", "HARD DISK", "MULTIMEDIA DEVICES", "POWER SUPPLY", "DESIGN & FEATURES"]
lap_filter_to_category = ["HARD_DISK", "PRICE", "OPERATION_PERFORMANCE", "FANS_COOLING", "FANS_COOLING", "FANS_COOLING", "HARD_DISK", "MULTIMEDIA_DEVICES", "POWER_SUPPLY", "DESIGN_FEATURES"]


def parser_getting():
    parser = argparse.ArgumentParser(description='Bidirectional MRC-based sentiment triplet extraction')
    parser.add_argument('--task', type=int, default=3, choices=[2, 3])
    parser.add_argument('--domain', type=str, default='res', choices=['res', 'lap', 'hot', 'fin'])
    parser.add_argument('--language', type=str, default='eng', choices=['eng', 'zho'])

    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--log_path', type=str, default="./log/")
    parser.add_argument('--save_model_path', type=str, default="./model/")
    parser.add_argument('--output_path', type=str, default="./tasks/")
    parser.add_argument('--model_name', type=str, default="AOC")

    parser.add_argument('--train_data', type=str, default="eng_restaurant_train_alltasks.jsonl")
    parser.add_argument('--infer_data', type=str, default="eng_restaurant_dev_task2.jsonl")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "evaluate", "inference"])
    parser.add_argument('--max_len', type=str, default="max_len", choices=["max_len"])
    parser.add_argument('--max_aspect_num', type=str, default="max_aspect_num", choices=["max_aspect_num"])

    parser.add_argument('--reload', type=bool, default=False)

    # parser.add_argument('--bert_model_type', type=str, default="F:\\myhuggingface\\bert\\bert-base-multilingual-uncased")
    # parser.add_argument('--bert_model_type', type=str, default="bert-base-multilingual-uncased")
    parser.add_argument('--bert_model_type', type=str, default="/home/zhangyou/myhuggingface/bert/bert-base-multilingual-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.90)

    # training hyper-parameter
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--epoch_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)

    args = parser.parse_args()
    return args


def evaluate(args, model, tokenize, batch_generator, test_data, beta, logger, gpu, max_len):
    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_cate_target_num = 0

    triplet_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_cate_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_cate_match_num = 0

    for batch_index, batch_dict in enumerate(batch_generator):

        triplets_target = test_data[batch_index].triplet_list
        asp_target = test_data[batch_index].aspect_list
        opi_target = test_data[batch_index].opinion_list
        asp_opi_target = test_data[batch_index].asp_opi_list
        asp_cate_target = test_data[batch_index].asp_cate_list

        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_cate_predict = []

        forward_pair_list = []
        backward_pair_list = []

        forward_pair_prob = []
        backward_pair_prob = []

        forward_pair_ind_list = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []

        final_asp_ind_list = []
        final_opi_ind_list = []

        ok_start_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()

        ok_start_tokens = batch_dict['forward_asp_query'][0][ok_start_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 'A')

        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)

        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []

        for start_index in range(f_asp_start_ind.size(0)):
            if batch_dict['forward_asp_answer_start'][0, start_index] != -1:
                if f_asp_start_ind[start_index].item() == 1:
                    f_asp_start_index_temp.append(start_index)
                    f_asp_start_prob_temp.append(f_asp_start_prob[start_index].item())
                if f_asp_end_ind[start_index].item() == 1:
                    f_asp_end_index_temp.append(start_index)
                    f_asp_end_prob_temp.append(f_asp_end_prob[start_index].item())
        f_asp_start_index, f_asp_end_index, f_asp_prob = Utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, max_len)

        for start_index in range(len(f_asp_start_index)):
            opinion_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1):
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(tokenize.convert_tokens_to_ids('?'))
            opinion_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query)

            opinion_query = torch.tensor(opinion_query).long()
            if gpu:
                opinion_query = opinion_query.cuda()
            opinion_query = torch.cat([opinion_query, ok_start_tokens], -1).unsqueeze(0)
            opinion_query_seg += [1] * ok_start_tokens.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().unsqueeze(0)
            if gpu:
                opinion_query_mask = opinion_query_mask.cuda()
            opinion_query_seg = torch.tensor(opinion_query_seg).long().unsqueeze(0)
            if gpu:
                opinion_query_seg = opinion_query_seg.cuda()

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 'AO')

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

            f_opi_start_index, f_opi_end_index, f_opi_prob = Utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, max_len)

            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in
                       range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[start_index] - 5, f_asp_end_index[start_index] - 5]
                opi_ind = [f_opi_start_index[idx] - f_opi_length, f_opi_end_index[idx] - f_opi_length]
                # TODO
                temp_prob = math.sqrt(f_asp_prob[start_index] * f_opi_prob[idx])
                if asp_ind + opi_ind not in forward_pair_list:
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 'O')
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for start_index in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, start_index] != -1:
                if b_opi_start_ind[start_index].item() == 1:
                    b_opi_start_index_temp.append(start_index)
                    b_opi_start_prob_temp.append(b_opi_start_prob[start_index].item())
                if b_opi_end_ind[start_index].item() == 1:
                    b_opi_end_index_temp.append(start_index)
                    b_opi_end_prob_temp.append(b_opi_end_prob[start_index].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = Utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp, max_len)

        for start_index in range(len(b_opi_start_index)):
            aspect_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(tokenize.convert_tokens_to_ids('describe'))
            aspect_query.append(tokenize.convert_tokens_to_ids('?'))
            aspect_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long()
            if gpu:
                aspect_query = aspect_query.cuda()
            aspect_query = torch.cat([aspect_query, ok_start_tokens], -1).unsqueeze(0)
            aspect_query_seg += [1] * ok_start_tokens.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().unsqueeze(0)
            if gpu:
                aspect_query_mask = aspect_query_mask.cuda()
            aspect_query_seg = torch.tensor(aspect_query_seg).long().unsqueeze(0)
            if gpu:
                aspect_query_seg = aspect_query_seg.cuda()

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 'OA')

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = Utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, max_len)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                opi_ind = [b_opi_start_index[start_index] - 5, b_opi_end_index[start_index] - 5]
                # TODO
                temp_prob = math.sqrt(b_asp_prob[idx] * b_opi_prob[start_index])
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list or forward_pair_prob[idx] >= beta:
                if forward_pair_list[idx][0] not in final_asp_list:
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else:
                    asp_index = final_asp_list.index(forward_pair_list[idx][0])
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
        # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])

        # category
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            for idy in range(predict_opinion_num):
                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])

                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)


                if args.task == 3 and 'category_query' in batch_dict:
                    category_query = tokenize.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                         '[CLS] What category given the aspect'.split(' ')])
                    category_query += final_asp_list[idx]
                    category_query += tokenize.convert_tokens_to_ids(
                        [word.lower() for word in 'and the opinion'.split(' ')])
                    category_query += final_opi_list[idx][idy]
                    category_query.append(tokenize.convert_tokens_to_ids('?'))
                    category_query.append(tokenize.convert_tokens_to_ids('[SEP]'))

                    category_query_seg = [0] * len(category_query)
                    category_query = torch.tensor(category_query).long()
                    if gpu:
                        category_query = category_query.cuda()
                    category_query = torch.cat([category_query, ok_start_tokens], -1).unsqueeze(0)
                    category_query_seg += [1] * ok_start_tokens.size(0)
                    category_query_mask = torch.ones(category_query.size(1)).float().unsqueeze(0)
                    if gpu:
                        category_query_mask = category_query_mask.cuda()
                    category_query_seg = torch.tensor(category_query_seg).long().unsqueeze(0)
                    if gpu:
                        category_query_seg = category_query_seg.cuda()

                    category_scores = model(category_query, category_query_mask, category_query_seg, 'C')
                    category_predicted = torch.argmax(category_scores[0], dim=0).item()
                    if asp_f + [category_predicted] not in asp_cate_predict:
                        asp_cate_predict.append(asp_f + [category_predicted])
                    triplet_predict = asp_f + opi_f + [category_predicted]
                    if triplet_predict not in triplets_predict:
                        triplets_predict.append(triplet_predict)
                else:
                    category_predicted = None
                    if asp_f + [category_predicted] not in asp_cate_predict:
                        asp_cate_predict.append(asp_f + [category_predicted])
                    triplet_predict = asp_f + opi_f + [category_predicted]
                    if triplet_predict not in triplets_predict:
                        triplets_predict.append(triplet_predict)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_cate_target_num += len(asp_cate_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_cate_predict_num += len(asp_cate_predict)

        for trip in triplets_predict:
            for trip_ in triplets_target:
                if trip_ == trip:
                    triplet_match_num += 1

        with open('./task1&2_predict.txt', 'a') as f:
            f.write(f"{triplets_predict}\n")

        for trip in asp_predict:
            for trip_ in asp_target:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_predict:
            for trip_ in opi_target:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_predict:
            for trip_ in asp_opi_target:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_cate_predict:
            for trip_ in asp_cate_target:
                if trip_ == trip:
                    asp_cate_match_num += 1

    precision = float(triplet_match_num) / float(triplet_predict_num + 1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))

    precision_aspect = float(asp_match_num) / float(asp_predict_num + 1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num + 1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect + 1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num + 1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num + 1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion + 1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_category = float(asp_cate_match_num) / float(asp_cate_predict_num + 1e-6)
    recall_aspect_category = float(asp_cate_match_num) / float(asp_cate_target_num + 1e-6)
    f1_aspect_category = 2 * precision_aspect_category * recall_aspect_category / (
            precision_aspect_category + recall_aspect_category + 1e-6)
    logger.info('Aspect-Category - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_category,
                                                                             recall_aspect_category,
                                                                             f1_aspect_category))

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num + 1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num + 1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion + 1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    return f1


def inference(args, model, tokenize, batch_generator, beta, logger, gpu, max_len, category_mapping):
    ids_to_categories = [key for key, value in sorted(category_mapping.items(), key=lambda item: item[1])]
    model.eval()
    output_data_triple = []
    output_data_quadra = []
    for batch_index, batch_dict in enumerate(batch_generator):
        dump_data_triple = {
            "ID": batch_dict['id'][0],
            "Triplet": [],
        }
        dump_data_quadra = {
            "ID": batch_dict['id'][0],
            "Quadruplet": [],
        }
        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_cate_predict = []

        forward_pair_list = []
        backward_pair_list = []

        forward_pair_prob = []
        backward_pair_prob = []

        forward_pair_ind_list = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []

        final_asp_ind_list = []
        final_opi_ind_list = []

        ok_start_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()

        ok_start_tokens = batch_dict['forward_asp_query'][0][ok_start_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 'A')

        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)

        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []

        for start_index in range(f_asp_start_ind.size(0)):
            if batch_dict['forward_asp_answer_start'][0, start_index] != -1:
                if f_asp_start_ind[start_index].item() == 1:
                    f_asp_start_index_temp.append(start_index)
                    f_asp_start_prob_temp.append(f_asp_start_prob[start_index].item())
                if f_asp_end_ind[start_index].item() == 1:
                    f_asp_end_index_temp.append(start_index)
                    f_asp_end_prob_temp.append(f_asp_end_prob[start_index].item())
        f_asp_start_index, f_asp_end_index, f_asp_prob = Utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, max_len)

        for start_index in range(len(f_asp_start_index)):
            opinion_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1):
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(tokenize.convert_tokens_to_ids('?'))
            opinion_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query)

            opinion_query = torch.tensor(opinion_query).long()
            if gpu:
                opinion_query = opinion_query.cuda()
            opinion_query = torch.cat([opinion_query, ok_start_tokens], -1).unsqueeze(0)
            opinion_query_seg += [1] * ok_start_tokens.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().unsqueeze(0)
            if gpu:
                opinion_query_mask = opinion_query_mask.cuda()
            opinion_query_seg = torch.tensor(opinion_query_seg).long().unsqueeze(0)
            if gpu:
                opinion_query_seg = opinion_query_seg.cuda()

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 'AO')

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

            f_opi_start_index, f_opi_end_index, f_opi_prob = Utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, max_len)

            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in
                       range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[start_index] - 5, f_asp_end_index[start_index] - 5]
                opi_ind = [f_opi_start_index[idx] - f_opi_length, f_opi_end_index[idx] - f_opi_length]
                # TODO
                temp_prob = math.sqrt(f_asp_prob[start_index] * f_opi_prob[idx])
                if asp_ind + opi_ind not in forward_pair_list:
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 'O')
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for start_index in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, start_index] != -1:
                if b_opi_start_ind[start_index].item() == 1:
                    b_opi_start_index_temp.append(start_index)
                    b_opi_start_prob_temp.append(b_opi_start_prob[start_index].item())
                if b_opi_end_ind[start_index].item() == 1:
                    b_opi_end_index_temp.append(start_index)
                    b_opi_end_prob_temp.append(b_opi_end_prob[start_index].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = Utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp, max_len)

        for start_index in range(len(b_opi_start_index)):
            aspect_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(tokenize.convert_tokens_to_ids('describe'))
            aspect_query.append(tokenize.convert_tokens_to_ids('?'))
            aspect_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long()
            if gpu:
                aspect_query = aspect_query.cuda()
            aspect_query = torch.cat([aspect_query, ok_start_tokens], -1).unsqueeze(0)
            aspect_query_seg += [1] * ok_start_tokens.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().unsqueeze(0)
            if gpu:
                aspect_query_mask = aspect_query_mask.cuda()
            aspect_query_seg = torch.tensor(aspect_query_seg).long().unsqueeze(0)
            if gpu:
                aspect_query_seg = aspect_query_seg.cuda()

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 'OA')

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = Utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, max_len)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                opi_ind = [b_opi_start_index[start_index] - 5, b_opi_end_index[start_index] - 5]
                # TODO
                temp_prob = math.sqrt(b_asp_prob[idx] * b_opi_prob[start_index])
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list or forward_pair_prob[idx] >= beta:
                if forward_pair_list[idx][0] not in final_asp_list:
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else:
                    asp_index = final_asp_list.index(forward_pair_list[idx][0])
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
        # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])

        # category
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            for idy in range(predict_opinion_num):

                # for category
                if args.task == 3:
                    category_query = tokenize.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                         '[CLS] What category given the aspect'.split(' ')])
                    category_query += final_asp_list[idx]
                    category_query += tokenize.convert_tokens_to_ids(
                        [word.lower() for word in 'and the opinion'.split(' ')])
                    category_query += final_opi_list[idx][idy]
                    category_query.append(tokenize.convert_tokens_to_ids('?'))
                    category_query.append(tokenize.convert_tokens_to_ids('[SEP]'))

                    category_query_seg = [0] * len(category_query)
                    category_query = torch.tensor(category_query).long()
                    if gpu:
                        category_query = category_query.cuda()
                    category_query = torch.cat([category_query, ok_start_tokens], -1).unsqueeze(0)
                    category_query_seg += [1] * ok_start_tokens.size(0)
                    category_query_mask = torch.ones(category_query.size(1)).float().unsqueeze(0)
                    if gpu:
                        category_query_mask = category_query_mask.cuda()
                    category_query_seg = torch.tensor(category_query_seg).long().unsqueeze(0)
                    if gpu:
                        category_query_seg = category_query_seg.cuda()

                    category_scores = model(category_query, category_query_mask, category_query_seg, 'C')
                    category_predicted = torch.argmax(category_scores[0], dim=0).item()
                else:
                    category_predicted = None

                # for valence predictor
                valence_query = tokenize.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What valence given the aspect'.split(' ')])
                valence_query += final_asp_list[idx]
                valence_query += tokenize.convert_tokens_to_ids(
                    [word.lower() for word in 'and the opinion'.split(' ')])
                valence_query += final_opi_list[idx][idy]
                valence_query.append(tokenize.convert_tokens_to_ids('?'))
                valence_query.append(tokenize.convert_tokens_to_ids('[SEP]'))

                valence_query_seg = [0] * len(valence_query)
                valence_query = torch.tensor(valence_query).long()
                if gpu:
                    valence_query = valence_query.cuda()
                valence_query = torch.cat([valence_query, ok_start_tokens], -1).unsqueeze(0)
                valence_query_seg += [1] * ok_start_tokens.size(0)
                valence_query_mask = torch.ones(valence_query.size(1)).float().unsqueeze(0)
                if gpu:
                    valence_query_mask = valence_query_mask.cuda()
                valence_query_seg = torch.tensor(valence_query_seg).long().unsqueeze(0)
                if gpu:
                    valence_query_seg = valence_query_seg.cuda()

                valence_scores = model(valence_query, valence_query_mask, valence_query_seg, 'Valence')

                # for Arousal predictor
                arousal_query = tokenize.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What arousal given the aspect'.split(' ')])
                arousal_query += final_asp_list[idx]
                arousal_query += tokenize.convert_tokens_to_ids(
                    [word.lower() for word in 'and the opinion'.split(' ')])
                arousal_query += final_opi_list[idx][idy]
                arousal_query.append(tokenize.convert_tokens_to_ids('?'))
                arousal_query.append(tokenize.convert_tokens_to_ids('[SEP]'))

                arousal_query_seg = [0] * len(arousal_query)
                arousal_query = torch.tensor(arousal_query).long()
                if gpu:
                    arousal_query = arousal_query.cuda()
                arousal_query = torch.cat([arousal_query, ok_start_tokens], -1).unsqueeze(0)
                arousal_query_seg += [1] * ok_start_tokens.size(0)
                arousal_query_mask = torch.ones(arousal_query.size(1)).float().unsqueeze(0)
                if gpu:
                    arousal_query_mask = arousal_query_mask.cuda()
                arousal_query_seg = torch.tensor(arousal_query_seg).long().unsqueeze(0)
                if gpu:
                    arousal_query_seg = arousal_query_seg.cuda()

                arousal_scores = model(arousal_query, arousal_query_mask, arousal_query_seg, 'Arousal')

                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [category_predicted] not in asp_cate_predict:
                    asp_cate_predict.append(asp_f + [category_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)

                triplet_predict = asp_f + opi_f + [category_predicted] + [str(round(valence_scores.item(), 2))] + [str(round(arousal_scores.item(), 2))]
                if triplet_predict not in triplets_predict:
                    triplets_predict.append(triplet_predict)

        # with open('./task1&2_predict.txt', 'a') as f:
        #     f.write(f"{triplets_predict}\n")
        # print(str(batch_index).center(20, "="))
        # print(batch_dict['id'])
        # print(batch_dict['line'])
        word_list_ids = batch_dict['forward_asp_query'][0][5:]
        # print(tokenize.convert_ids_to_tokens(word_list_ids))
        for triplet in triplets_predict:
            # print(triplet)
            # print(tokenize.decode(word_list_ids[triplet[0]:triplet[1]+1]),
            # tokenize.decode(word_list_ids[triplet[2]:triplet[3]+1]),
            # ids_to_categories[triplet[4]] if triplet[4] is not None else None, sep=',')

            meta_triplet = {}
            meta_triplet["Aspect"] = tokenize.decode(word_list_ids[triplet[0]:triplet[1] + 1])
            meta_triplet["Opinion"] = tokenize.decode(word_list_ids[triplet[2]:triplet[3]+1])
            meta_triplet["VA"] = triplet[5] + "#" + triplet[6]
            if args.language in ['zho']:
                meta_triplet["Aspect"] = meta_triplet["Aspect"].replace(" ", "")
                meta_triplet["Opinion"] = meta_triplet["Opinion"].replace(" ", "")
            dump_data_triple['Triplet'].append(meta_triplet)
            if args.task == 3:
                meta_quadra = {}
                meta_quadra["Aspect"] = meta_triplet["Aspect"]
                meta_quadra['Category'] = ids_to_categories[triplet[4]]
                meta_quadra["Opinion"] = meta_triplet["Opinion"]
                meta_quadra["VA"] = meta_triplet["VA"]
                dump_data_quadra['Quadruplet'].append(meta_quadra)

        output_data_triple.append(dump_data_triple)
        output_data_quadra.append(dump_data_quadra)

    out_put_file_task2_name = args.output_path + "subtask_2/" + out_put_file_name_map[args.domain + '_' + args.language]
    with open(out_put_file_task2_name, 'w', encoding='utf-8') as f:
        for i, item in enumerate(output_data_triple):
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str+'\n')

    if args.task == 3:
        out_put_file_task3_name = args.output_path + "subtask_3/" + out_put_file_name_map[args.domain + '_' + args.language]
        with open(out_put_file_task3_name, 'w', encoding='utf-8') as f:
            for i, item in enumerate(output_data_quadra):
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + '\n')


def train(args, train_total_data, test_total_data, inference_dataset, category_mapping):
    log_path = args.log_path + args.model_name + '.log'
    model_path = args.save_model_path + 'task' + str(args.task) + '_' + args.domain + '_' + args.language + '.pth'

    # init logger and tokenize
    logger, fh, sh = Utils.get_logger(log_path)
    tokenize = AutoTokenizer.from_pretrained(args.bert_model_type)

    # for training
    train_data = train_total_data['train']
    max_len = train_total_data[args.max_len]
    max_aspect_num = train_total_data[args.max_aspect_num]

    # for evaluating as inference text
    dev_data = train_total_data['dev']

    # for evaluating as golden labels
    dev_standard = test_total_data['dev']

    model = DimABSA(args.hidden_size, args.bert_model_type, len(category_mapping))
    if args.gpu:
        model = model.cuda()

    if args.mode == 'evaluate':
        test_dataset = ReviewDataset(args, dev_data)
        # load checkpoint
        logger.info('loading model......')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])

        # eval
        logger.info('evaluating......')

        batch_generator_test = generate_batches(dataset=test_dataset, batch_size=1, shuffle=False, gpu=args.gpu)
        evaluate(args, model, tokenize, batch_generator_test, dev_standard, args.inference_beta, logger, args.gpu, max_len)

    if args.mode == 'inference':
        ID_list, Text_list, QA_list = inference_dataset
        inf_dataset = InferenceReviewDataset(args, QA_list)
        # load checkpoint
        logger.info('loading model......')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])

        # eval
        logger.info('evaluating......')

        batch_generator_test = generate_batches(dataset=inf_dataset, batch_size=1, shuffle=False,
                                                gpu=args.gpu)

        inference(args, model, tokenize, batch_generator_test, args.inference_beta, logger, args.gpu, max_len, category_mapping)

    elif args.mode == 'train':
        train_dataset = ReviewDataset(args, train_data)
        dev_dataset = ReviewDataset(args, dev_data)
        batch_num_train = train_dataset.get_batch_num(args.batch_size)

        # optimizer
        logger.info('initial optimizer......')
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "bert" not in n],
             'lr': args.learning_rate, 'weight_decay': 0.01}]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_bert_rate, correct_bias=False)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_bert_rate)

        # load saved model, optimizer and epoch num
        if args.reload and os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
        else:
            start_epoch = 1
            logger.info('New model and optimizer from epoch 1')

        # scheduler
        training_steps = args.epoch_num * batch_num_train
        warmup_steps = int(training_steps * args.warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)

        # training
        logger.info('begin training......')
        best_f1 = 0.

        for epoch in range(start_epoch, args.epoch_num + 1):
            model.train()
            model.zero_grad()
            batch_generator = generate_batches(dataset=train_dataset, batch_size=args.batch_size,
                                                    gpu=args.gpu)

            for batch_index, batch_dict in enumerate(batch_generator):

                optimizer.zero_grad()

                f_aspect_start_scores, f_aspect_end_scores = model(batch_dict['forward_asp_query'],
                                                                   batch_dict['forward_asp_query_mask'],
                                                                   batch_dict['forward_asp_query_seg'], 'A')
                f_asp_loss = Utils.calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                                                         batch_dict['forward_asp_answer_start'],
                                                         batch_dict['forward_asp_answer_end'], args.gpu)

                b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                             batch_dict['backward_opi_query_mask'],
                                                             batch_dict['backward_opi_query_seg'], 'O')
                b_opi_loss = Utils.calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                                                         batch_dict['backward_opi_answer_start'],
                                                         batch_dict['backward_opi_answer_end'], args.gpu)

                # TODO f_opi
                batch_dict['forward_opi_query'] = batch_dict['forward_opi_query'] \
                    .view(-1, batch_dict['forward_opi_query'].size(-1))
                batch_dict['forward_opi_query_mask'] = batch_dict['forward_opi_query_mask'] \
                    .view(-1, batch_dict['forward_opi_query_mask'].size(-1))
                batch_dict['forward_opi_query_seg'] = batch_dict['forward_opi_query_seg'] \
                    .view(-1, batch_dict['forward_opi_query_seg'].size(-1))

                f_opi_start_scores, f_opi_end_scores = model(
                    batch_dict['forward_opi_query'],
                    batch_dict['forward_opi_query_mask'],
                    batch_dict['forward_opi_query_seg'],
                    'AO')

                f_opi_loss = Utils.calculate_entity_loss(
                    f_opi_start_scores, f_opi_end_scores,
                    batch_dict['forward_opi_answer_start'].view(-1, batch_dict[
                        'forward_opi_answer_start'].size(-1)),
                    batch_dict['forward_opi_answer_end'].view(-1, batch_dict[
                        'forward_opi_answer_end'].size(-1)), args.gpu
                ) / max_aspect_num

                # TODO b_asp
                batch_dict['backward_asp_query'] = batch_dict['backward_asp_query'] \
                    .view(-1, batch_dict['backward_asp_query'].size(-1))
                batch_dict['backward_asp_query_mask'] = batch_dict['backward_asp_query_mask'] \
                    .view(-1, batch_dict['backward_asp_query_mask'].size(-1))
                batch_dict['backward_asp_query_seg'] = batch_dict['backward_asp_query_seg'] \
                    .view(-1, batch_dict['backward_asp_query_seg'].size(-1))

                b_asp_start_scores, b_asp_end_scores = model(
                    batch_dict['backward_asp_query'],
                    batch_dict['backward_asp_query_mask'],
                    batch_dict['backward_asp_query_seg'],
                    'OA')

                b_asp_loss = Utils.calculate_entity_loss(
                    b_asp_start_scores, b_asp_end_scores,
                    batch_dict['backward_asp_answer_start'].view(-1, batch_dict[
                        'backward_asp_answer_start'].size(-1)),
                    batch_dict['backward_asp_answer_end'].view(-1, batch_dict[
                        'backward_asp_answer_end'].size(-1)), args.gpu
                ) / max_aspect_num

                if args.task == 3 and 'category_query' in batch_dict:
                    # TODO category
                    batch_dict['category_query'] = batch_dict['category_query'] \
                        .view(-1, batch_dict['category_query'].size(-1))
                    batch_dict['category_query_mask'] = batch_dict['category_query_mask'] \
                        .view(-1, batch_dict['category_query_mask'].size(-1))
                    batch_dict['category_query_seg'] = batch_dict['category_query_seg'] \
                        .view(-1, batch_dict['category_query_seg'].size(-1))

                    category_scores = model(
                        batch_dict['category_query'],
                        batch_dict['category_query_mask'],
                        batch_dict['category_query_seg'],
                        'C')

                    category_loss = Utils.calculate_category_loss(
                        category_scores,
                        batch_dict['category_answer'].view(-1)) / max_aspect_num
                else:
                    category_loss = torch.tensor(0.).to("cuda")


                # TODO Valence and Arousal
                batch_dict['valence_query'] = batch_dict['valence_query'].view(-1, batch_dict['valence_query'].size(-1))
                batch_dict['valence_query_mask'] = batch_dict['valence_query_mask'].view(-1, batch_dict['valence_query_mask'].size(-1))
                batch_dict['valence_query_seg'] = batch_dict['valence_query_seg'].view(-1, batch_dict['valence_query_seg'].size(-1))

                valence_scores = model(batch_dict['valence_query'],
                                       batch_dict['valence_query_mask'],
                                       batch_dict['valence_query_seg'], 'Valence')

                valence_loss = Utils.calculate_valence_loss(
                    valence_scores,
                    batch_dict['valence_answer'].view(-1)) / max_aspect_num

                batch_dict['arousal_query'] = batch_dict['arousal_query'] \
                    .view(-1, batch_dict['arousal_query'].size(-1))
                batch_dict['arousal_query_mask'] = batch_dict['arousal_query_mask'] \
                    .view(-1, batch_dict['arousal_query_mask'].size(-1))
                batch_dict['arousal_query_seg'] = batch_dict['arousal_query_seg'] \
                    .view(-1, batch_dict['arousal_query_seg'].size(-1))

                arousal_scores = model(batch_dict['arousal_query'],
                                       batch_dict['arousal_query_mask'],
                                       batch_dict['arousal_query_seg'], 'Arousal')

                arousal_loss = Utils.calculate_arousal_loss(
                    arousal_scores,
                    batch_dict['arousal_answer'].view(-1)) / max_aspect_num

                # loss
                loss_sum = f_asp_loss + f_opi_loss + b_opi_loss + b_asp_loss + \
                           args.beta * category_loss + valence_loss*0.2 + arousal_loss*0.2
                loss_sum.backward()
                optimizer.step()
                scheduler.step()

                # train logger
                if (batch_index + 1) % 10 == 0:
                    logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\tforward Loss:{};{}\t'
                                ' backward Loss:{};{}\t Sentiment Loss:{}\tValence Loss:{}\t Arousal Loss:{}\t'.
                                format(epoch, args.epoch_num, batch_index + 1, batch_num_train,
                                       round(loss_sum.item(), 4),
                                       round(f_asp_loss.item(), 4), round(f_opi_loss.item(), 4),
                                       round(b_opi_loss.item(), 4), round(b_asp_loss.item(), 4),
                                       round(category_loss.item(), 4), round(valence_loss.item(), 4),
                                       round(arousal_loss.item(), 4))
                                )

            # validation
            batch_generator_dev = generate_batches(dataset=dev_dataset, batch_size=1, shuffle=False, gpu=args.gpu)
            logger.info("dev")
            dev_f1 = evaluate(args, model, tokenize, batch_generator_dev, dev_standard, args.inference_beta, logger,
                          args.gpu, max_len)

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                logger.info('Model saved after epoch {}'.format(epoch))
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, model_path)

        # do inference
        ID_list, Text_list, QA_list = inference_dataset
        inf_dataset = InferenceReviewDataset(args, QA_list)
        logger.info('loading model......')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        logger.info('inference......')
        batch_generator_test = generate_batches(dataset=inf_dataset, batch_size=1, shuffle=False,
                                                gpu=args.gpu)
        inference(args, model, tokenize, batch_generator_test, args.inference_beta, logger, args.gpu, max_len, category_mapping)

    else:
        logger.info('Error mode!')
        exit(1)
    logger.removeHandler(fh)
    logger.removeHandler(sh)


def load_inference_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_type)
    inference_datasets = []

    # train_data_path, dev_data_path, test_data_path = dataset_path_map[args.domain + '_' + args.language]
    inference_data_path = args.data_path + args.infer_data
    category_dict, category_list = category_map[args.domain]

    with open(inference_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_id = data['ID']
            text = data['Text'].lower()
            text = " ".join(tokenizer.tokenize(text))
            inference_datasets.append((data_id, text))

    inference_dataset = dataset_inference_process(args, inference_datasets, category_dict, tokenizer)

    return inference_dataset


def load_train_data_multilingual(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_type)
    def find_word_indices(text, phrase):
        words = tokenizer.tokenize(text)[:256]
        if phrase == "NULL" or not phrase:
            return " ".join(words), "NULL", -1, -1
        phrase_words = tokenizer.tokenize(phrase)

        for i in range(len(words) - len(phrase_words) + 1):
            if words[i:i + len(phrase_words)] == phrase_words:
                return " ".join(words), " ".join(phrase_words), i, i + len(phrase_words) - 1
        return " ".join(words), " ".join(phrase_words), -1, -1

    train_datasets = {
        'train': [],
        'dev': [],
    }

    # train_data_path, dev_data_path, test_data_path = dataset_path_map[args.domain + '_' + args.language]

    train_data_path = args.data_path + args.train_data

    category_dict, category_list = category_map[args.domain]

    all_data =[]
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)

    random.seed(42)
    random.shuffle(all_data)

    # splitting training dataset for new_training dataset and development data
    total_count = len(all_data)
    train_count = int(total_count * 0.8)

    for i, data in enumerate(all_data):
        text = data['Text']
        quadruplets = data['Quadruplet']
        quintuplets = []
        for quad in quadruplets:
            if 'Category' in quad and args.task == 3:
                category = quad['Category'].upper()
                if args.domain in ['lap']:
                    replacement_dict = dict(zip(lap_filter_from_category, lap_filter_to_category))
                    category = replace_using_dict(category, replacement_dict)
                assert category in category_list, print(f"Wrong category type {category} in dataset, need filter.")
            else:
                category = None

            new_text, new_opinion, a_start, a_end = find_word_indices(text, quad['Aspect'].lower())
            _, new_opinion, o_start, o_end = find_word_indices(text, quad['Opinion'].lower())

            va_parts = quad['VA'].split('#')
            valence = va_parts[0]
            arousal = va_parts[1] if len(va_parts) > 1 else "0.0"

            quint = (
                [a_start, a_end],
                [o_start, o_end],
                category,
                valence,
                arousal
            )
            quintuplets.append(quint)

        quint_str = ", ".join([str(q) for q in quintuplets])
        output_line = f"{new_text}####[{quint_str}]"
        if i < train_count:
            dataset_type = 'train'
        else:
            dataset_type = 'dev'
        train_datasets[dataset_type].append(output_line)

    train_dataset, eval_dataset = dataset_process(args, train_datasets, category_dict, tokenizer)

    return train_dataset, eval_dataset, category_dict

if __name__ == '__main__':
    args = parser_getting()
    create_directory(args)
    train_dataset, test_dataset, category_dict = load_train_data_multilingual(args)
    inference_dataset = load_inference_data(args) # ID_LIST, TEXT_LIST, QA_LIST
    train(args, train_dataset, test_dataset, inference_dataset, category_dict)
