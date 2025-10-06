import random
import re
import Utils as Data
from transformers import AutoTokenizer


dataset_type_list = ["train", "dev"]

triplet_pattern = re.compile(r'[(](.*?)[)]', re.S)  # 匹配圆括号 () 中的内容
aspect_and_opinion_pattern = re.compile(r'[\[](.*?)[]]', re.S)  # 匹配方括号 [] 中的内容
category_pattern = re.compile(r"['](.*?)[']", re.S)

forward_aspect_query_template = ["[CLS]", "what", "aspects", "?", "[SEP]"]
forward_opinion_query_template = ["[CLS]", "what", "opinion", "given", "the", "aspect", "?", "[SEP]"]
backward_opinion_query_template = ["[CLS]", "what", "opinions", "?", "[SEP]"]
backward_aspect_query_template = ["[CLS]", "what", "aspect", "does", "the", "opinion", "describe", "?", "[SEP]"]
category_query_template = ["[CLS]", "what", "category", "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]
valence_query_template = ["[CLS]", "what", "valence", "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]
arousal_query_template = ["[CLS]", "what", "arousal", "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]


def print_QA(QA: Data.QueryAndAnswer, tokenizer):
    print('*' * 100)
    print('line:', QA.line, '\n',
          'forward_asp_query:', ids_to_tokens(QA.forward_asp_query, tokenizer), '\n',
          'forward_opi_query:', ids_to_tokens(QA.forward_opi_query,tokenizer), '\n',
          'forward_asp_query_mask:', QA.forward_asp_query_mask, '\n',
          'forward_asp_query_seg:', QA.forward_asp_query_seg, '\n',
          'forward_opi_query_mask:', QA.forward_opi_query_mask, '\n',
          'forward_opi_query_seg:', QA.forward_opi_query_seg, '\n',
          'forward_asp_query_start:', QA.forward_asp_answer_start, '\n',
          'forward_asp_query_end:', QA.forward_asp_answer_end, '\n',
          'forward_opi_query_start:', QA.forward_opi_answer_start, '\n',
          'forward_opi_query_end:', QA.forward_opi_answer_end, '\n',

          'backward_asp_query:', ids_to_tokens(QA.backward_asp_query, tokenizer), '\n',
          'backward_opi_query:', ids_to_tokens(QA.backward_opi_query, tokenizer), '\n',
          'backward_asp_query_mask:', QA.backward_asp_query_mask, '\n',
          'backward_asp_query_seg:', QA.backward_asp_query_seg, '\n',
          'backward_opi_query_mask:', QA.backward_opi_query_mask, '\n',
          'backward_opi_query_seg:', QA.backward_opi_query_seg, '\n',
          'backward_asp_answer_start:', QA.backward_asp_answer_start, '\n',
          'backward_asp_answer_end:', QA.backward_asp_answer_end, '\n',
          'backward_opi_answer_start:', QA.backward_opi_answer_start, '\n',
          'backward_opi_answer_end:', QA.backward_opi_answer_end, '\n',

          'category_query:', ids_to_tokens(QA.category_query, tokenizer), '\n',
          'category_answer:', QA.category_answer, '\n',
          'category_query_mask:', QA.category_query_mask, '\n',
          'category_query_seg:', QA.category_query_seg, '\n',

          'valence_query:', ids_to_tokens(QA.valence_query, tokenizer), '\n',
          'valence_answer:', QA.valence_answer, '\n',
          'valence_query_mask:', QA.valence_query_mask, '\n',
          'valence_query_seg:', QA.valence_query_seg, '\n',

          'arousal_query:', ids_to_tokens(QA.arousal_query, tokenizer), '\n',
          'arousal_answer:', QA.arousal_answer, '\n',
          'arousal_query_mask:', QA.arousal_query_mask, '\n',
          'arousal_query_seg:', QA.arousal_query_seg, '\n'
          )

    print(QA.line)
    token_list = ids_to_tokens(QA.forward_asp_query, tokenizer)
    for i in range(len(token_list)):
        if QA.forward_asp_answer_start[i] == 1:
            print('forward asp start:', token_list[i])
        if QA.forward_asp_answer_end[i] == 1:
            print('forward asp end:', token_list[i])

    token_list = ids_to_tokens(QA.forward_opi_query, tokenizer)
    for i in range(len(token_list)):
        for j in range(len(token_list[i])):
            if QA.forward_opi_answer_start[i][j] == 1:
                print('forward opi start:', token_list[i][j])
            if QA.forward_opi_answer_end[i][j] == 1:
                print('forward opi end:', token_list[i][j])

    token_list = ids_to_tokens(QA.backward_opi_query, tokenizer)
    for i in range(len(token_list)):
        if QA.backward_opi_answer_start[i] == 1:
            print('backward opi start:', token_list[i])
        if QA.backward_opi_answer_end[i] == 1:
            print('backward opi end:', token_list[i])

    token_list = ids_to_tokens(QA.backward_asp_query, tokenizer)
    for i in range(len(token_list)):
        for j in range(len(token_list[i])):
            if QA.backward_asp_answer_start[i][j] == 1:
                print('backward asp start:', token_list[i][j])
            if QA.backward_asp_answer_end[i][j] == 1:
                print('backward asp end:', token_list[i][j])

    token_list = ids_to_tokens(QA.category_query, tokenizer)
    for i in range(len(token_list)):
        print('category[{}]:'.format(i), QA.category_answer[i])

    token_list = ids_to_tokens(QA.valence_query, tokenizer)
    for i in range(len(token_list)):
        print('valence[{}]:'.format(i), QA.valence_answer[i])

    token_list = ids_to_tokens(QA.arousal_query, tokenizer)
    for i in range(len(token_list)):
        print('arousal[{}]:'.format(i), QA.arousal_answer[i])

    print('*' * 100)


def valid(QA: Data.QueryAndAnswer):
    assert len(QA.forward_asp_query) == len(QA.forward_asp_answer_start) \
           == len(QA.forward_asp_answer_end) == len(QA.forward_asp_query_mask) \
           == len(QA.forward_asp_query_seg)
    for i in range(len(QA.forward_opi_query)):
        assert len(QA.forward_opi_query[i]) == len(QA.forward_opi_answer_start[i]) \
               == len(QA.forward_opi_answer_end[i]) == len(QA.forward_opi_query_mask[i]) \
               == len(QA.forward_opi_query_seg[i])

    assert len(QA.backward_opi_query) == len(QA.backward_opi_answer_start) \
           == len(QA.backward_opi_answer_end) == len(QA.backward_opi_query_mask) \
           == len(QA.backward_opi_query_seg)
    for i in range(len(QA.backward_asp_query)):
        assert len(QA.backward_asp_query[i]) == len(QA.backward_asp_answer_start[i]) \
               == len(QA.backward_asp_answer_end[i]) == len(QA.backward_asp_query_mask[i]) \
               == len(QA.backward_asp_query_seg[i])

    assert len(QA.category_query) == len(QA.category_answer) == len(QA.category_query_mask) \
           == len(QA.category_query_seg)

    for i in range(len(QA.category_query)):
        if QA.category_query[i] is not None:
            assert len(QA.category_query[i]) == len(QA.category_query_mask[i]) == len(QA.category_query_seg[i])

    assert len(QA.valence_query) == len(QA.valence_answer) == len(QA.valence_query_mask) \
           == len(QA.valence_query_seg)
    for i in range(len(QA.valence_query)):
        assert len(QA.valence_query[i]) == len(QA.valence_query_mask[i]) == len(QA.valence_query_seg[i])

    assert len(QA.arousal_query) == len(QA.arousal_answer) == len(QA.arousal_query_mask) \
           == len(QA.arousal_query_seg)
    for i in range(len(QA.arousal_query)):
        assert len(QA.arousal_query[i]) == len(QA.arousal_query_mask[i]) == len(QA.arousal_query_seg[i])


# ids to tokens
def ids_to_tokens(input_ids_list, tokenizer):
    if not isinstance(input_ids_list[0], list):
        return tokenizer.convert_ids_to_tokens(input_ids_list)
    token_list = []
    for input_ids in input_ids_list:
        token_list.append(tokenizer.convert_ids_to_tokens(input_ids))
    return token_list

# tokens to ids
def tokens_to_ids(QA_list, tokenizer):
    max_len = 0
    for QA in QA_list:
        QA.forward_asp_query = tokenizer.convert_tokens_to_ids(QA.forward_asp_query)
        if len(QA.forward_asp_query) > max_len:
            max_len = len(QA.forward_asp_query)
        for i in range(len(QA.forward_opi_query)):
            QA.forward_opi_query[i] = tokenizer.convert_tokens_to_ids(QA.forward_opi_query[i])
            if len(QA.forward_opi_query[i]) > max_len:
                max_len = len(QA.forward_opi_query[i])

        QA.backward_opi_query = tokenizer.convert_tokens_to_ids(QA.backward_opi_query)
        if len(QA.backward_opi_query) > max_len:
            max_len = len(QA.backward_opi_query)
        for i in range(len(QA.backward_asp_query)):
            QA.backward_asp_query[i] = tokenizer.convert_tokens_to_ids(QA.backward_asp_query[i])
            if len(QA.backward_asp_query[i]) > max_len:
                max_len = len(QA.backward_asp_query[i])

        for i in range(len(QA.category_query)):
            if QA.category_query[i] is not None:
                QA.category_query[i] = tokenizer.convert_tokens_to_ids(QA.category_query[i])
                if len(QA.category_query[i]) > max_len:
                    max_len = len(QA.category_query[i])

        for i in range(len(QA.valence_query)):
            QA.valence_query[i] = tokenizer.convert_tokens_to_ids(QA.valence_query[i])
            if len(QA.valence_query[i]) > max_len:
                max_len = len(QA.valence_query[i])

        for i in range(len(QA.arousal_query)):
            QA.arousal_query[i] = tokenizer.convert_tokens_to_ids(QA.arousal_query[i])
            if len(QA.arousal_query[i]) > max_len:
                max_len = len(QA.arousal_query[i])

        valid(QA)
    return QA_list, max_len


def list_to_object(dataset_object):
    line = []
    forward_asp_query = []
    forward_opi_query = []
    forward_asp_query_mask = []
    forward_asp_query_seg = []
    forward_opi_query_mask = []
    forward_opi_query_seg = []
    forward_asp_answer_start = []
    forward_asp_answer_end = []
    forward_opi_answer_start = []
    forward_opi_answer_end = []

    backward_asp_query = []
    backward_opi_query = []
    backward_asp_query_mask = []
    backward_asp_query_seg = []
    backward_opi_query_mask = []
    backward_opi_query_seg = []
    backward_asp_answer_start = []
    backward_asp_answer_end = []
    backward_opi_answer_start = []
    backward_opi_answer_end = []

    category_query = []
    category_answer = []
    category_query_mask = []
    category_query_seg = []

    valence_query = []
    valence_answer = []
    valence_query_mask = []
    valence_query_seg = []

    arousal_query = []
    arousal_answer = []
    arousal_query_mask = []
    arousal_query_seg = []
    for QA in dataset_object:
        line.append(QA.line)
        forward_asp_query.append(QA.forward_asp_query)
        forward_opi_query.append(QA.forward_opi_query)
        forward_asp_query_mask.append(QA.forward_asp_query_mask)
        forward_asp_query_seg.append(QA.forward_asp_query_seg)
        forward_opi_query_mask.append(QA.forward_opi_query_mask)
        forward_opi_query_seg.append(QA.forward_opi_query_seg)
        forward_asp_answer_start.append(QA.forward_asp_answer_start)
        forward_asp_answer_end.append(QA.forward_asp_answer_end)
        forward_opi_answer_start.append(QA.forward_opi_answer_start)
        forward_opi_answer_end.append(QA.forward_opi_answer_end)

        backward_asp_query.append(QA.backward_asp_query)
        backward_opi_query.append(QA.backward_opi_query)
        backward_asp_query_mask.append(QA.backward_asp_query_mask)
        backward_asp_query_seg.append(QA.backward_asp_query_seg)
        backward_opi_query_mask.append(QA.backward_opi_query_mask)
        backward_opi_query_seg.append(QA.backward_opi_query_seg)
        backward_asp_answer_start.append(QA.backward_asp_answer_start)
        backward_asp_answer_end.append(QA.backward_asp_answer_end)
        backward_opi_answer_start.append(QA.backward_opi_answer_start)
        backward_opi_answer_end.append(QA.backward_opi_answer_end)

        category_query.append(QA.category_query)
        category_answer.append(QA.category_answer)
        category_query_mask.append(QA.category_query_mask)
        category_query_seg.append(QA.category_query_seg)

        valence_query.append(QA.valence_query)
        valence_answer.append(QA.valence_answer)
        valence_query_mask.append(QA.valence_query_mask)
        valence_query_seg.append(QA.valence_query_seg)

        arousal_query.append(QA.arousal_query)
        arousal_answer.append(QA.arousal_answer)
        arousal_query_mask.append(QA.arousal_query_mask)
        arousal_query_seg.append(QA.arousal_query_seg)

    return Data.QueryAndAnswer(line=line,
                               forward_asp_query=forward_asp_query,
                               forward_opi_query=forward_opi_query,
                               forward_asp_query_mask=forward_asp_query_mask,
                               forward_asp_query_seg=forward_asp_query_seg,
                               forward_opi_query_mask=forward_opi_query_mask,
                               forward_opi_query_seg=forward_opi_query_seg,
                               forward_asp_answer_start=forward_asp_answer_start,
                               forward_asp_answer_end=forward_asp_answer_end,
                               forward_opi_answer_start=forward_opi_answer_start,
                               forward_opi_answer_end=forward_opi_answer_end,
                               backward_asp_query=backward_asp_query,
                               backward_opi_query=backward_opi_query,
                               backward_asp_query_mask=backward_asp_query_mask,
                               backward_asp_query_seg=backward_asp_query_seg,
                               backward_opi_query_mask=backward_opi_query_mask,
                               backward_opi_query_seg=backward_opi_query_seg,
                               backward_asp_answer_start=backward_asp_answer_start,
                               backward_asp_answer_end=backward_asp_answer_end,
                               backward_opi_answer_start=backward_opi_answer_start,
                               backward_opi_answer_end=backward_opi_answer_end,
                               category_query=category_query,
                               category_answer=category_answer,
                               category_query_mask=category_query_mask,
                               category_query_seg=category_query_seg,

                               valence_query=valence_query,
                               valence_answer=valence_answer,
                               valence_query_mask=valence_query_mask,
                               valence_query_seg=valence_query_seg,

                               arousal_query=arousal_query,
                               arousal_answer=arousal_answer,
                               arousal_query_mask=arousal_query_mask,
                               arousal_query_seg=arousal_query_seg
                               )


def dataset_align(dataset_object, max_tokens_len, max_aspect_num, tokenizer):
    for dataset_type in dataset_type_list:
        tokenized_QA_list = dataset_object[dataset_type]
        for tokenized_QA in tokenized_QA_list:

            tokenized_QA.forward_asp_query.extend([0] * (max_tokens_len - len(tokenized_QA.forward_asp_query)))
            tokenized_QA.forward_asp_query_mask.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_asp_query_mask)))
            tokenized_QA.forward_asp_query_seg.extend(
                [1] * (max_tokens_len - len(tokenized_QA.forward_asp_query_seg)))

            tokenized_QA.forward_asp_answer_start.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.forward_asp_answer_start)))
            tokenized_QA.forward_asp_answer_end.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.forward_asp_answer_end)))

            for i in range(len(tokenized_QA.forward_opi_query)):
                tokenized_QA.forward_opi_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_query[i])))
                tokenized_QA.forward_opi_answer_start[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.forward_opi_answer_start[i])))
                tokenized_QA.forward_opi_answer_end[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.forward_opi_answer_end[i])))

                tokenized_QA.forward_opi_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_query_mask[i])))
                tokenized_QA.forward_opi_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.forward_opi_query_seg[i])))

            tokenized_QA.backward_opi_query.extend([0] * (max_tokens_len - len(tokenized_QA.backward_opi_query)))

            tokenized_QA.backward_opi_query_mask.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opi_query_mask)))
            tokenized_QA.backward_opi_query_seg.extend(
                [1] * (max_tokens_len - len(tokenized_QA.backward_opi_query_seg)))

            tokenized_QA.backward_opi_answer_start.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.backward_opi_answer_start)))
            tokenized_QA.backward_opi_answer_end.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.backward_opi_answer_end)))

            for i in range(len(tokenized_QA.backward_asp_query)):
                tokenized_QA.backward_asp_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_query[i])))
                tokenized_QA.backward_asp_answer_start[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.backward_asp_answer_start[i])))
                tokenized_QA.backward_asp_answer_end[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.backward_asp_answer_end[i])))

                tokenized_QA.backward_asp_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_query_mask[i])))
                tokenized_QA.backward_asp_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.backward_asp_query_seg[i])))

            for i in range(len(tokenized_QA.category_query)):
                if tokenized_QA.category_query[i] is not None:
                    tokenized_QA.category_query[i].extend([0] * (max_tokens_len - len(tokenized_QA.category_query[i])))

                    tokenized_QA.category_query_mask[i].extend(
                        [0] * (max_tokens_len - len(tokenized_QA.category_query_mask[i])))
                    tokenized_QA.category_query_seg[i].extend(
                        [1] * (max_tokens_len - len(tokenized_QA.category_query_seg[i])))

            for i in range(len(tokenized_QA.valence_query)):
                tokenized_QA.valence_query[i].extend([0] * (max_tokens_len - len(tokenized_QA.valence_query[i])))
                tokenized_QA.valence_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.valence_query_mask[i])))
                tokenized_QA.valence_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.valence_query_seg[i])))

            for i in range(len(tokenized_QA.arousal_query)):
                tokenized_QA.arousal_query[i].extend([0] * (max_tokens_len - len(tokenized_QA.arousal_query[i])))
                tokenized_QA.arousal_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.arousal_query_mask[i])))
                tokenized_QA.arousal_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.arousal_query_seg[i])))

            for i in range(max_aspect_num - len(tokenized_QA.forward_opi_query)):
                tokenized_QA.forward_opi_query.insert(-1, tokenized_QA.forward_opi_query[0])
                tokenized_QA.forward_opi_query_mask.insert(-1, tokenized_QA.forward_opi_query_mask[0])
                tokenized_QA.forward_opi_query_seg.insert(-1, tokenized_QA.forward_opi_query_seg[0])

                tokenized_QA.forward_opi_answer_start.insert(-1, tokenized_QA.forward_opi_answer_start[0])
                tokenized_QA.forward_opi_answer_end.insert(-1, tokenized_QA.forward_opi_answer_end[0])

                tokenized_QA.backward_asp_query.insert(-1, tokenized_QA.backward_asp_query[0])
                tokenized_QA.backward_asp_query_mask.insert(-1, tokenized_QA.backward_asp_query_mask[0])
                tokenized_QA.backward_asp_query_seg.insert(-1, tokenized_QA.backward_asp_query_seg[0])

                tokenized_QA.backward_asp_answer_start.insert(-1, tokenized_QA.backward_asp_answer_start[0])
                tokenized_QA.backward_asp_answer_end.insert(-1, tokenized_QA.backward_asp_answer_end[0])

                tokenized_QA.category_query.insert(-1, tokenized_QA.category_query[0])
                tokenized_QA.category_query_mask.insert(-1, tokenized_QA.category_query_mask[0])
                tokenized_QA.category_query_seg.insert(-1, tokenized_QA.category_query_seg[0])
                tokenized_QA.category_answer.insert(-1, tokenized_QA.category_answer[0])

                tokenized_QA.valence_query.insert(-1, tokenized_QA.valence_query[0])
                tokenized_QA.valence_query_mask.insert(-1, tokenized_QA.valence_query_mask[0])
                tokenized_QA.valence_query_seg.insert(-1, tokenized_QA.valence_query_seg[0])
                tokenized_QA.valence_answer.insert(-1, tokenized_QA.valence_answer[0])

                tokenized_QA.arousal_query.insert(-1, tokenized_QA.arousal_query[0])
                tokenized_QA.arousal_query_mask.insert(-1, tokenized_QA.arousal_query_mask[0])
                tokenized_QA.arousal_query_seg.insert(-1, tokenized_QA.arousal_query_seg[0])
                tokenized_QA.arousal_answer.insert(-1, tokenized_QA.arousal_answer[0])

            valid(tokenized_QA)
            if random.random() == 0.999:
                print_QA(tokenized_QA, tokenizer)
    return dataset_object


def get_start_end(str_list):
    index_list = [int(s) for s in str_list]
    return [index_list[0] + 1, index_list[-1] + 1]


def make_QA(args, line, word_list, aspect_list, opinion_list, category_list, valence_list, arousal_list):
    # word_list.append("[SEP]")
    forward_asp_query = forward_aspect_query_template + word_list
    forward_asp_query_mask = [1] * len(forward_asp_query)
    forward_asp_query_seg = [0] * len(forward_aspect_query_template) + [1] * len(word_list)
    forward_asp_answer_start = [-1] * len(forward_aspect_query_template) + [0] * len(word_list)
    forward_asp_answer_end = [-1] * len(forward_aspect_query_template) + [0] * len(word_list)
    forward_opi_query = []
    forward_opi_query_mask = []
    forward_opi_query_seg = []
    forward_opi_answer_start = []
    forward_opi_answer_end = []

    backward_opi_query = backward_opinion_query_template + word_list
    backward_opi_query_mask = [1] * len(backward_opi_query)
    backward_opi_query_seg = [0] * len(backward_opinion_query_template) + [1] * len(word_list)
    backward_opi_answer_start = [-1] * len(backward_opinion_query_template) + [0] * len(word_list)
    backward_opi_answer_end = [-1] * len(backward_opinion_query_template) + [0] * len(word_list)
    backward_asp_query = []
    backward_asp_query_mask = []
    backward_asp_query_seg = []
    backward_asp_answer_start = []
    backward_asp_answer_end = []

    category_query = []
    category_query_mask = []
    category_query_seg = []
    category_answer = category_list
    category_word_list = word_list[:]
    category_query_mask_init = [1] * len(category_word_list)

    valence_query = []
    valence_query_mask = []
    valence_query_seg = []
    valence_answer = valence_list
    valence_word_list = word_list[:]
    valence_query_mask_init = [1] * len(valence_word_list)

    arousal_query = []
    arousal_query_mask = []
    arousal_query_seg = []
    arousal_answer = arousal_list
    arousal_word_list = word_list[:]
    arousal_query_mask_init = [1] * len(arousal_word_list)

    for i in range(len(aspect_list)):
        for aspect_index in range(aspect_list[i][0], aspect_list[i][1] + 1):
            category_word_list[aspect_index] = "[PAD]"
            category_query_mask_init[aspect_index] = 0
        for opinion_index in range(opinion_list[i][0], opinion_list[i][1] + 1):
            category_word_list[opinion_index] = "[PAD]"
            category_query_mask_init[opinion_index] = 0

    for i in range(len(aspect_list)):
        asp = aspect_list[i]
        opi = opinion_list[i]

        forward_asp_answer_start[len(forward_aspect_query_template) + asp[0]] = 1
        forward_asp_answer_end[len(forward_aspect_query_template) + asp[1]] = 1

        opi_query_temp = forward_opinion_query_template[0:6] + word_list[asp[0]:asp[1] + 1] + \
                         forward_opinion_query_template[6:] + word_list
        forward_opi_query.append(opi_query_temp)

        opi_query_mask_temp = [1] * len(opi_query_temp)
        opi_query_seg_temp = [0] * (len(opi_query_temp) - len(word_list)) + [1] * len(word_list)
        forward_opi_query_mask.append(opi_query_mask_temp)
        forward_opi_query_seg.append(opi_query_seg_temp)

        opi_answer_start_temp = [-1] * (len(opi_query_temp) - len(word_list)) + [0] * len(word_list)
        opi_answer_start_temp[len(opi_query_temp) - len(word_list) + opi[0]] = 1
        opi_answer_end_temp = [-1] * (len(opi_query_temp) - len(word_list)) + [0] * len(word_list)
        opi_answer_end_temp[len(opi_query_temp) - len(word_list) + opi[1]] = 1
        forward_opi_answer_start.append(opi_answer_start_temp)
        forward_opi_answer_end.append(opi_answer_end_temp)

        backward_opi_answer_start[len(backward_opinion_query_template) + opi[0]] = 1
        backward_opi_answer_end[len(backward_opinion_query_template) + opi[1]] = 1

        asp_query_temp = backward_aspect_query_template[0:6] + word_list[opi[0]:opi[1] + 1] + \
                         backward_aspect_query_template[6:] + word_list
        backward_asp_query.append(asp_query_temp)

        asp_query_mask_temp = [1] * len(asp_query_temp)
        asp_query_seg_temp = [0] * (len(asp_query_temp) - len(word_list)) + [1] * len(word_list)
        backward_asp_query_mask.append(asp_query_mask_temp)
        backward_asp_query_seg.append(asp_query_seg_temp)

        asp_answer_start_temp = [-1] * (len(asp_query_temp) - len(word_list)) + [0] * len(word_list)
        asp_answer_start_temp[len(asp_query_temp) - len(word_list) + asp[0]] = 1
        asp_answer_end_temp = [-1] * (len(asp_query_temp) - len(word_list)) + [0] * len(word_list)
        asp_answer_end_temp[len(asp_query_temp) - len(word_list) + asp[1]] = 1
        backward_asp_answer_start.append(asp_answer_start_temp)
        backward_asp_answer_end.append(asp_answer_end_temp)

        # for generating category queries, if task_3 anc category is not None, it is None.
        if args.task != 3 or category_list[0] is None:
            category_query.append(None)
            category_query_mask.append(None)
            category_query_mask_init.append(None)
            category_query_seg.append(None)
        else:
            category_word_list_temp = category_word_list[:]

            category_query_mask_init_temp = category_query_mask_init[:]
            for aspect_index in range(asp[0], asp[1] + 1):
                category_word_list_temp[aspect_index] = word_list[aspect_index]
                category_query_mask_init_temp[aspect_index] = 1

            for opinion_index in range(opi[0], opi[1] + 1):
                category_word_list_temp[opinion_index] = word_list[opinion_index]
                category_query_mask_init_temp[opinion_index] = 1

            category_query_temp = category_query_template[0:6] + word_list[asp[0]:asp[1] + 1] + \
                                  category_query_template[6:9] + word_list[opi[0]:opi[1] + 1] + \
                                  category_query_template[9:] + category_word_list_temp
            category_query.append(category_query_temp)
            category_query_mask_temp = [1] * (len(category_query_temp) - len(category_word_list_temp)) + \
                                       category_query_mask_init_temp
            category_query_seg_temp = [0] * (len(category_query_temp) - len(category_word_list_temp)) + \
                                      [1] * len(category_word_list_temp)
            category_query_mask.append(category_query_mask_temp)
            category_query_seg.append(category_query_seg_temp)

        valence_word_list_temp = valence_word_list[:]
        valence_query_mask_init_temp = valence_query_mask_init[:]
        valence_query_temp = valence_query_template[0:6] + word_list[asp[0]:asp[1] + 1] + \
                             valence_query_template[6:9] + word_list[opi[0]:opi[1] + 1] + \
                             valence_query_template[9:] + valence_word_list_temp
        valence_query_mask_temp = [1] * (len(valence_query_temp) - len(valence_word_list_temp)) + \
                                  valence_query_mask_init_temp
        valence_query_seg_temp = [0] * (len(valence_query_temp) - len(valence_word_list_temp)) + \
                                 [1] * len(valence_word_list_temp)
        valence_query.append(valence_query_temp)
        valence_query_mask.append(valence_query_mask_temp)
        valence_query_seg.append(valence_query_seg_temp)

        arousal_word_list_temp = arousal_word_list[:]
        arousal_query_mask_init_temp = arousal_query_mask_init[:]
        arousal_query_temp = arousal_query_template[0:6] + word_list[asp[0]:asp[1] + 1] + \
                             arousal_query_template[6:9] + word_list[opi[0]:opi[1] + 1] + \
                             arousal_query_template[9:] + arousal_word_list_temp
        arousal_query_mask_temp = [1] * (len(arousal_query_temp) - len(arousal_word_list_temp)) + \
                                  arousal_query_mask_init_temp
        arousal_query_seg_temp = [0] * (len(arousal_query_temp) - len(arousal_word_list_temp)) + \
                                 [1] * len(arousal_word_list_temp)
        arousal_query.append(arousal_query_temp)
        arousal_query_mask.append(arousal_query_mask_temp)
        arousal_query_seg.append(arousal_query_seg_temp)

    return Data.QueryAndAnswer(line=line,
                               forward_asp_query=forward_asp_query,
                               forward_opi_query=forward_opi_query,
                               forward_asp_query_mask=forward_asp_query_mask,
                               forward_asp_query_seg=forward_asp_query_seg,
                               forward_opi_query_mask=forward_opi_query_mask,
                               forward_opi_query_seg=forward_opi_query_seg,
                               forward_asp_answer_start=forward_asp_answer_start,
                               forward_asp_answer_end=forward_asp_answer_end,
                               forward_opi_answer_start=forward_opi_answer_start,
                               forward_opi_answer_end=forward_opi_answer_end,

                               backward_asp_query=backward_asp_query,
                               backward_opi_query=backward_opi_query,
                               backward_asp_query_mask=backward_asp_query_mask,
                               backward_asp_query_seg=backward_asp_query_seg,
                               backward_opi_query_mask=backward_opi_query_mask,
                               backward_opi_query_seg=backward_opi_query_seg,
                               backward_asp_answer_start=backward_asp_answer_start,
                               backward_asp_answer_end=backward_asp_answer_end,
                               backward_opi_answer_start=backward_opi_answer_start,
                               backward_opi_answer_end=backward_opi_answer_end,

                               category_query=category_query,
                               category_answer=category_answer,
                               category_query_mask=category_query_mask,
                               category_query_seg=category_query_seg,

                               valence_query=valence_query,
                               valence_answer=valence_answer,
                               valence_query_mask=valence_query_mask,
                               valence_query_seg=valence_query_seg,

                               arousal_query=arousal_query,
                               arousal_answer=arousal_answer,
                               arousal_query_mask=arousal_query_mask,
                               arousal_query_seg=arousal_query_seg,
                               )


def make_inference_QA(args, text_id, line, word_list):
    # word_list.append("[SEP]")
    forward_asp_query = forward_aspect_query_template + word_list
    forward_asp_query_mask = [1] * len(forward_asp_query)
    forward_asp_query_seg = [0] * len(forward_aspect_query_template) + [1] * len(word_list)
    forward_asp_answer_start = [-1] * len(forward_aspect_query_template) + [0] * len(word_list)
    forward_asp_answer_end = [-1] * len(forward_aspect_query_template) + [0] * len(word_list)

    backward_opi_query = backward_opinion_query_template + word_list
    backward_opi_query_mask = [1] * len(backward_opi_query)
    backward_opi_query_seg = [0] * len(backward_opinion_query_template) + [1] * len(word_list)
    backward_opi_answer_start = [-1] * len(backward_opinion_query_template) + [0] * len(word_list)
    backward_opi_answer_end = [-1] * len(backward_opinion_query_template) + [0] * len(word_list)


    return Data.Query(text_id=text_id,
                      line=line,
                      forward_asp_query=forward_asp_query,
                      forward_asp_query_mask=forward_asp_query_mask,
                      forward_asp_query_seg=forward_asp_query_seg,
                      forward_asp_answer_start=forward_asp_answer_start,
                      forward_asp_answer_end=forward_asp_answer_end,
                      backward_opi_query=backward_opi_query,
                      backward_opi_query_mask=backward_opi_query_mask,
                      backward_opi_query_seg=backward_opi_query_seg,
                      backward_opi_answer_start=backward_opi_answer_start,
                      backward_opi_answer_end=backward_opi_answer_end,
                      )


def line_data_process(args, line, category_mapping, isQA=True):
    # Line sample:
    # judging from previous posts this used to be a good place , but not any longer .####[([10, 10], [13, 15], 'RESTAURANT#GENERAL', '3.62', '5.88')]
    split = line.split("####")
    assert len(split) == 2

    max_aspect_num = 0
    max_len = 0
    word_list = split[0].split()
    word_list.insert(0, "null")

    triplet_str_list = re.findall(triplet_pattern, split[1])
    aspect_list = [re.findall(aspect_and_opinion_pattern, triplet)[0] for triplet in triplet_str_list]
    aspect_list = [get_start_end(aspect.split(', ')) for aspect in aspect_list]
    if len(aspect_list) > max_aspect_num:
        max_aspect_num = len(aspect_list)
    opinion_list = [re.findall(aspect_and_opinion_pattern, triplet)[1] for triplet in triplet_str_list]
    opinion_list = [get_start_end(opinion.split(', ')) for opinion in opinion_list]

    if args.task != 3:
        category_list = [None for _ in triplet_str_list]
    else:
        category_list = [category_mapping[re.findall(category_pattern, triplet)[0]] for triplet in triplet_str_list]

    valence_list = [eval(triplet.split(',')[-2].strip().strip('"').strip("'")) for triplet in triplet_str_list]
    arousal_list = [eval(triplet.split(',')[-1].strip().strip('"').strip("'")) for triplet in triplet_str_list]

    assert len(aspect_list) > 0 and len(opinion_list) > 0 and len(valence_list) > 0 and len(arousal_list) > 0
    assert len(aspect_list) == len(opinion_list) == len(category_list) == len(valence_list) == len(aspect_list)

    for i in range(len(aspect_list)):
        if (aspect_list[i][1] - aspect_list[i][0] + 1) > max_len:
            max_len = aspect_list[i][1] - aspect_list[i][0] + 1
        if (opinion_list[i][1] - opinion_list[i][0] + 1) > max_len:
            max_len = opinion_list[i][1] - opinion_list[i][0] + 1
    if isQA:
        return make_QA(args, line, word_list, aspect_list, opinion_list, category_list, valence_list,
                       arousal_list), max_aspect_num, max_len
    else:
        return line, aspect_list, opinion_list, category_list, valence_list, arousal_list


def line_inference_data_process(args, text_id, line, isQA=True):
    # Line sample:
    # judging from previous posts this used to be a good place , but not any longer .####[([10, 10], [13, 15], 'RESTAURANT#GENERAL', '3.62', '5.88')]

    # line is a pure text for inference
    split = line
    word_list = split.split()
    word_list = [word.lower() for word in word_list]
    word_list.insert(0, "null")
    if isQA:
        return make_inference_QA(args, text_id, line, word_list)
    else:
        return line


def train_data_process(args, text, category_mapping):
    QA_list = []
    max_aspect_num = 0
    max_len = 0
    for line in text[:]:
        QA, max_aspect_temp, max_len_temp = line_data_process(args, line, category_mapping)
        if max_aspect_temp > max_aspect_num:
            max_aspect_num = max_aspect_temp
        if max_len_temp > max_len:
            max_len = max_len_temp
        QA_list.append(QA)
    return QA_list, max_aspect_num, max_len


def inference_data_process(args, text):
    QA_list = []
    TEXT_list = []
    ID_list = []
    for line in text[:]:
        ID_list.append(line[0])
        TEXT_list.append(line[1])
        QA = line_inference_data_process(args, line[0], line[1])
        QA_list.append(QA)
    return ID_list, TEXT_list, QA_list


def test_data_process(args, text, category_mapping):
    test_dataset = []
    for line in text[:]:
        (line, aspect_list_temp, opinion_list_temp, category_list_temp,
         valence_list_temp, arousal_list_temp) = line_data_process(args, line, category_mapping, isQA=False)
        aspect_list = []
        opinion_list = []
        asp_opi_list = []
        asp_cate_list = []
        triplet_list = []
        valence_list = []
        arousal_list = []
        VA_list = []
        for i in range(0, len(aspect_list_temp)):
            if aspect_list_temp[i] not in aspect_list:
                aspect_list.append(aspect_list_temp[i])
            if opinion_list_temp[i] not in opinion_list:
                opinion_list.append(opinion_list_temp[i])
            asp_opi_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], opinion_list_temp[i][0],
                            opinion_list_temp[i][1]]
            asp_cate_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], category_list_temp[i]]
            triplet_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], opinion_list_temp[i][0],
                            opinion_list_temp[i][1], category_list_temp[i]]
            valence_temp = [valence_list_temp[i]]
            arousal_temp = [arousal_list_temp[i]]
            VA_temp = [valence_list_temp[i], arousal_list_temp[i]]
            asp_opi_list.append(asp_opi_temp)
            if asp_cate_temp not in asp_cate_list:
                asp_cate_list.append(asp_cate_temp)
            triplet_list.append(triplet_temp)
            valence_list.append(valence_temp)
            arousal_list.append(arousal_temp)
            VA_list.append(VA_temp)
        test_dataset.append(Data.TestDataset(
            line=line,
            aspect_list=aspect_list,
            opinion_list=opinion_list,
            asp_opi_list=asp_opi_list,
            asp_cate_list=asp_cate_list,
            triplet_list=triplet_list,
            valence_list=valence_list,
            arousal_list=arousal_list,
            VA_list=VA_list
        ))
    return test_dataset


def dataset_process(args, datatsets, category_mapping, tokenizer):
    train_dataset_object = {}
    test_dataset_object = {}
    max_tokens_len = 0
    max_aspect_num = 0
    max_len = 0
    for dataset_type in dataset_type_list:
        text_lines = datatsets[dataset_type]
        QA_list, max_aspect_temp, max_len_temp = train_data_process(args, text_lines, category_mapping)
        train_dataset_object[dataset_type], max_tokens_temp = tokens_to_ids(QA_list, tokenizer)
        test_dataset_object[dataset_type] = test_data_process(args, text_lines, category_mapping)
        if max_tokens_temp > max_tokens_len:
            max_tokens_len = max_tokens_temp
        if max_aspect_temp > max_aspect_num:
            max_aspect_num = max_aspect_temp
        if max_len_temp > max_len:
            max_len = max_len_temp
    train_dataset_object = dataset_align(train_dataset_object, max_tokens_len, max_aspect_num+1, tokenizer)
    train_dataset_object['max_tokens_len'] = max_tokens_len
    train_dataset_object['max_aspect_num'] = max_aspect_num
    train_dataset_object['max_len'] = max_len

    print('Max length of training tokens: ', max_tokens_len)
    print('Max length in aspect/opinion: ', max_len)
    print('Maximum aspect/opinion in a single sample: ', max_aspect_num)
    return train_dataset_object, test_dataset_object


def dataset_inference_process(args, datasets, category_mapping, tokenizer):

    ID_list, TEXT_list, QA_list = inference_data_process(args, datasets)

    max_len = 0
    for QA in QA_list:
        QA.forward_asp_query = tokenizer.convert_tokens_to_ids(QA.forward_asp_query)
        if len(QA.forward_asp_query) > max_len:
            max_len = len(QA.forward_asp_query)
        QA.backward_opi_query = tokenizer.convert_tokens_to_ids(QA.backward_opi_query)
        if len(QA.backward_opi_query) > max_len:
            max_len = len(QA.backward_opi_query)
    return ID_list, TEXT_list, QA_list