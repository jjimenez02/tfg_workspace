import matplotlib.pyplot as plt


def plot_score(
        list_on_dicted_results,
        hyper_parameters_pair_names,
        model_name,
        inverse=False,
        mode='normal',
        in_same_graphic=False,
        accuracy_mode='accuracy'):
    if inverse:
        title = hyper_parameters_pair_names[0]
        x_label = hyper_parameters_pair_names[1]
        index_on_x_label = 1
        index_on_title = 0
    else:
        title = hyper_parameters_pair_names[1]
        x_label = hyper_parameters_pair_names[0]
        index_on_x_label = 0
        index_on_title = 1

    final_results_list = []
    for dicted_results in list_on_dicted_results:
        final_results_list.extend(list(dicted_results.items()))

    sorted_final_results_list =\
        sorted(final_results_list, key=lambda x: x[0][index_on_x_label])

    all_predefined_values = __get_unique_values_on_index(
        [x[0] for x in sorted_final_results_list], index_on_title)

    if in_same_graphic:
        __plot_in_same_graph(
            all_predefined_values,
            sorted_final_results_list,
            index_on_title,
            index_on_x_label,
            mode,
            x_label,
            title,
            accuracy_mode,
            model_name
        )
    else:
        __plot_in_different_graphs(
            all_predefined_values,
            sorted_final_results_list,
            index_on_title,
            index_on_x_label,
            mode,
            x_label,
            title,
            accuracy_mode,
            model_name
        )


def __get_unique_values_on_index(list_of_values, index):
    return {val[index] for val in list_of_values}


def __plot_in_same_graph(
        all_predefined_values,
        sorted_final_results_list,
        index_on_title,
        index_on_x_label,
        mode,
        x_label,
        title,
        accuracy_mode,
        model_name):
    for predefined_value in all_predefined_values:
        x_axis = []
        y_axis = []
        for result in sorted_final_results_list:
            if result[0][index_on_title] != predefined_value:
                continue
            result_hyper_params = result[0]
            result_scores = result[1]
            x_axis.append(result_hyper_params[index_on_x_label])
            if accuracy_mode == 'accuracy':
                current_result_score = result_scores[mode]
            elif accuracy_mode == 'loss':
                current_result_score = 1-result_scores[mode]
            else:
                raise Exception(
                    'you must specify a valid accuracy_mode:\
                        {\'accuracy\', \'loss\'}')
            y_axis.append(current_result_score)
        plt.plot(x_axis, y_axis, label=title + ': ' + str(predefined_value))

    plt.legend(loc=(1.04, 0))
    plt.xlabel(x_label)
    plt.ylabel('{}'.format(accuracy_mode.capitalize()))
    plt.title('{}\'s {} with predefined {}'.format(
        model_name, accuracy_mode, title))
    plt.show()


def __plot_in_different_graphs(
        all_predefined_values,
        sorted_final_results_list,
        index_on_title,
        index_on_x_label,
        mode,
        x_label,
        title,
        accuracy_mode,
        model_name):
    for predefined_value in all_predefined_values:
        x_axis = []
        y_axis = []
        for result in sorted_final_results_list:
            if result[0][index_on_title] != predefined_value:
                continue
            result_hyper_params = result[0]
            result_scores = result[1]
            x_axis.append(result_hyper_params[index_on_x_label])
            if accuracy_mode == 'accuracy':
                current_result_score = result_scores[mode]
            elif accuracy_mode == 'loss':
                current_result_score = 1-result_scores[mode]
            else:
                raise Exception(
                    'you must specify a valid accuracy_mode:\
                        {\'accuracy\', \'loss\'}')
            y_axis.append(current_result_score)

        plt.plot(x_axis, y_axis)
        plt.xlabel(x_label)
        plt.ylabel('{}'.format(accuracy_mode.capitalize()))
        plt.title('{}\'s {} with predefined {}={}'.format(
            model_name, accuracy_mode, title, predefined_value))
        plt.show()