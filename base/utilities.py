def format_dictionary(dictionary: dict):
    """short method to format dicts into a semi-tabular structure"""
    formatted_str = ''
    for x in dictionary:
        formatted_str += '\t{:20s}: {}\n'.format(x, dictionary[x])
    return formatted_str[:-1]


def calculate_tensor_size_after_convs(input_size: int, sizes: list, strides: list):
    """helper method to calculate output size of an input into a conv net consisting of conv layers with filter `sizes`
     and `strides`"""
    t = input_size
    for size, stride in zip(sizes, strides):
        t = int((t - size) / stride + 1)
    return t
