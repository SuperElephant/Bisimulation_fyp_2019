import standard_bisim.test_cases_generator as cg
import ml_algorithm.ml_algorithm as ml_a
import math
import numpy


def experiment(c_type, nodes_num, edge_type_num, case_number, graph_density, p_rate):
    file_name = c_type + "_n" + str(nodes_num) + "_e" + str(edge_type_num) + \
                "_" + str(case_number) + "_gd" + str(round(graph_density,3)) + "_p" + str(round(p_rate,3))
    print("training #%d: %s" % (1, file_name))
    cg.test_cases_generator(c_type=c_type,
                            number=case_number,
                            file_name=file_name,
                            min_node_number=nodes_num,
                            edge_type_number=edge_type_num,
                            probability=graph_density,
                            p_rate=p_rate)
    trainer = ml_a.ML(learning_rate=0.1,
                      epochs=500,
                      batch_size=100,
                      data_path="./data/" + file_name + ".csv",
                      model_name=file_name)
    trainer.fc()


if __name__ == '__main__':
    nodes_num = 5
    edge_type_num = 3

    # deep test
    graphs_sum = math.pow(2, nodes_num * nodes_num * edge_type_num)

    # print test report

    # width test
    c_type = "random"
    case_number = 10000

    p_rate = 0.5
    graph_density =0.5

    print("========== wide experiments ==========")
    graph_density_s = numpy.arange(0.1,1,0.1)
    print("---- ARGUMENTS ----\n"
          "nodes_num = %s \n"
          "edge_type_num = %s \n"
          "c_type = %s \n"
          "case_number = %s \n"
          "graph_density = %s\n"
          "p_rate = %s \n"
          "-------------------"
          % (nodes_num, edge_type_num, c_type, case_number, graph_density_s, p_rate))

    for graph_density in graph_density_s:
        experiment(c_type, nodes_num, edge_type_num, case_number, graph_density, p_rate)

    print("========== deep experiments ==========")
    nodes_num_s = range(5,15)
    graph_density =0.5
    print("---- ARGUMENTS ----\n"
          "nodes_num = %s \n"
          "edge_type_num = %s \n"
          "c_type = %s \n"
          "case_number = %s \n"
          "graph_density = %s\n"
          "p_rate = %s \n"
          "-------------------"
          % (nodes_num, edge_type_num, c_type, case_number, graph_density, p_rate))
    for nodes_num in nodes_num_s:
        experiment(c_type, nodes_num, edge_type_num, case_number, graph_density, p_rate)


