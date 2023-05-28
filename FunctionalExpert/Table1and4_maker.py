
import torch

def sum_tool(data : dict, index) :
    Sum = 0.0
    for item_data in data.values() :
        Sum += item_data[index]
    return Sum / len(data)

print("Proportion of (sub-)Functional Experts to All Experts")
for model in ("switchT5", "T5") :
    print("model = {}".format("T5" if model == "T5" else "Switch Transformer"))
    for concept in ("semantic", "knowledge", "task") :
        subfunction_y, function_y = 0.0, 0.0
        for layer in (range(1, 12, 2) if model == "switchT5" else range(12)) :
            function_data = torch.load("HypothesisTesting/{}/{}/function_{}.bin".format(concept, model, layer))
            subfunction_data = torch.load("HypothesisTesting/{}/{}/sub-function_{}.bin".format(concept, model, layer))
            subfunction_y += sum_tool(subfunction_data, 0) / (6 if model == "switchT5" else 12)
            function_y += function_data[0] / (6 if model == "switchT5" else 12)
        
        print("    concept = {}".format(concept))

        random_proportion = 0.0
        for layer in (range(1, 12, 2) if model == "switchT5" else range(12)) :
            random_proportion += torch.load("HypothesisTesting4RandomPartitioning/{}/{}/{}.bin".format(concept, model, layer))[0] / (6 if model == "switchT5" else 12)
        print("        function_expert = {}   function_random = {}".format(function_y, random_proportion))

        print("        sub-function_expert = {}   sub-function_random = {}".format(subfunction_y, 0.001))
        
        print("\n")

print("\n\n\n")

print("Modularization Degree")
for model in ("switchT5", "T5") :
    print("model = {}".format("T5" if model == "T5" else "Switch Transformer"))
    for concept in ("semantic", "knowledge", "task") :
        subfunction_y, function_y = 0.0, 0.0
        for layer in (range(1, 12, 2) if model == "switchT5" else range(12)) :
            function_data = torch.load("HypothesisTesting/{}/{}/function_{}.bin".format(concept, model, layer))
            subfunction_data = torch.load("HypothesisTesting/{}/{}/sub-function_{}.bin".format(concept, model, layer))
            subfunction_y += sum_tool(subfunction_data, 1) / (6 if model == "switchT5" else 12)
            function_y += function_data[1] / (6 if model == "switchT5" else 12)
        
        print("    concept = {}".format(concept))

        random_degree = 0.0
        for layer in (range(1, 12, 2) if model == "switchT5" else range(12)) :
            random_degree += torch.load("HypothesisTesting4RandomPartitioning/{}/{}/{}.bin".format(concept, model, layer))[1] / (6 if model == "switchT5" else 12)
        print("        function_expert = {}   function_random = {}".format(function_y, random_degree))
        
        print("        sub-function_expert = {}   sub-function_random = {}".format(subfunction_y, (0.02053333435058593 if model == "T5" else 0.022328309464454653)))
        
        print("\n")