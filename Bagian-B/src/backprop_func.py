import numpy as np

def d_relu(v: float, net: float):
    print(0 if net < 0 else v)
    return 0 if net < 0 else v

def d_relu_vect(v: np.ndarray, net: np.ndarray):
    res = np.empty_like(v)
    for row_idx, row in enumerate(v):
        for col_idx, val in enumerate(row):
            res[row_idx][col_idx] = 0 if net[row_idx][col_idx] < 0 else val
    return res

def d_softmax(o: float, net: float):
    return 1-o if net == 1.0 else -o

def d_softmax_vect(o: np.ndarray, target: np.ndarray):
    res = np.empty_like(o)
    for row_idx, row in enumerate(o):
        for col_idx, o_val in enumerate(row):
            res[row_idx][col_idx] = 1-o_val if target[col_idx] == 1.0 else -o_val
    return res

def d_sigmoid_vect(sigma_w_delta: np.ndarray, outputs: np.ndarray):
    res = np.copy(sigma_w_delta)
    for row_idx, row in enumerate(outputs):
        for col_idx, o_val in enumerate(row):
            res[row_idx][col_idx] *= o_val * (1-o_val)
    return res

"""
All variables output, target, nets, layer_input are in the form of
single column matrix
"""

def delta_linear_output(output: np.ndarray, target: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)

    return (target_mat - output_mat)

def delta_relu_output(output: np.ndarray, target: np.ndarray, nets: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)
    nets_mat = np.transpose(nets)

    res = d_relu_vect(target_mat - output_mat, nets_mat)

    return res

def delta_sigmoid_output(output: np.ndarray, target: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)
    
    return ((target_mat - output_mat) * output_mat * (1 - output_mat))

def delta_softmax_output(output: np.ndarray, target: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)

    return d_softmax_vect(output_mat, target_mat)

def delta_linear_hidden(
    ds_delta: np.ndarray, ds_w: np.ndarray
):
    sigma_delta_w = ds_delta * ds_w
    sigma_list = np.sum(sigma_delta_w, axis=1).transpose()

    return np.array([sigma_list[1:]])

def delta_relu_hidden(
    nets: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray
):
    sigma_delta_w = ds_delta * ds_w
    sigma_list = np.sum(sigma_delta_w, axis=1).transpose()
    sigma_nets = np.array([sigma_list[1:]])

    nets_mat = np.transpose(nets)
    res = d_relu_vect(sigma_nets, nets_mat)

    return res


def delta_sigmoid_hidden(
    output: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray
):
    sigma_delta_w = ds_delta * ds_w
    sigma_list = np.sum(sigma_delta_w, axis=1).transpose()
    sigma_nets = np.array([sigma_list[1:]])

    res = d_sigmoid_vect(sigma_nets, output)

    return res

# TODO fix
def delta_softmax_hidden(
    output: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray
):
    sigma_delta_w = ds_delta * ds_w
    sigma_list = np.sum(sigma_delta_w, axis=1).transpose()
    sigma_delta_w_nets = np.array([sigma_list[1:]])

    o_list = np.transpose(output).tolist()[0]

    output_deltas = []
    for i, oi in enumerate(o_list):
        z = []
        for j, oj in enumerate(o_list):
            if i == j:
                z.append(oi * (1 - oj))
            else:
                z.append(-(oi * oj))
        z_mat = np.array([z]).transpose()
        res_mat = z_mat @ sigma_delta_w_nets
        output_deltas.append(res_mat[0][0])

    return np.array([output_deltas])
