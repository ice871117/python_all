import time


def print_duration(before):
    duration = time.time() - before
    sec = int(duration + 0.5)
    min = int(sec / 60)
    sec = sec % 60
    hour = int(min / 60)
    min = min % 60
    print("time cost is %02d:%02d:%02d" % (hour, min, sec))


def get_moving_avg(old_value, new_value, ratio=0.98):
    if old_value > 0:
        return ratio * old_value + (1 - ratio) * new_value
    else:
        return new_value


def top_n_value(arr, n):
    sorted_dict = dict()
    for index, item in enumerate(arr):
        sorted_dict[item] = index
    sorted_keys = sorted(sorted_dict.keys(), reverse=True)
    result = []
    for index, key in enumerate(sorted_keys):
        if index >= n:
            break
        result.append(sorted_dict.get(key))
    return result


def trans_index_to_label(data_source, arr):
    if len(arr) > 0:
        res_str = ""
        for loop_index, item in enumerate(arr):
            if loop_index > 0:
                res_str += " "
            res_str += data_source.find_label_by_index(item)
        print("top_n_value=", res_str)
        return res_str
    else:
        return ""