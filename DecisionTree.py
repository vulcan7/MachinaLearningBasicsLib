import numpy as np
import pandas as pd
def train_test_split(df, test_size):
    
    #if isinstance(test_size, float):
    #test_size = int(round(test_size * len(df)))

    indices = df.index.tolist()
    np.random.seed(0)
    test_size = int(round(test_size * len(df)))

    test_indices = np.random.randint(low =0, high = len(indices), size=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
def classify_data(data, criteria = "mean"):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    if(criteria == "mean"):
        total_sum=0
        for i in range(len(unique_classes)):
            total_sum += counts_unique_classes[i]*i
            
        find_mean = int(round(total_sum/len(label_column)))
        classification = unique_classes[find_mean]
    else:
        find_median = int(round(len(label_column)/2))
        sum1 =0
        for i in range(len(unique_classes)):
                sum1 += counts_unique_classes[i]
                if(find_median <= sum1):
                    classification = unique_classes[i]

       # index = counts_unique_classes.argmax()
        
    
    return classification
def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):          # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy
def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy
def count_values(rows):
    #will return a dictionary with species values as key and frequency as values
    count={}
    #takes whole dataset in as argument
    for row in  rows:
        #traverse on each datapoint
        label=row[-1]
        #labels are in the last column
        #if label is not even once come initialise it
        if label not in count:
            count[label]=0
        #increase the count of present label by 1
        count[label]+=1
    return count
def calculate_gini(rows):
    #stores dictionary of frequency of labels
    count=count_values(rows)
    #initialise impurity as 1
    impurity=1
    for label in count:
        #probablity of a unique label
        probab_of_label=count[label]/float(len(rows))
        #calculation gini impurity acc to formula
        impurity-=probab_of_label**2
    return impurity
def calculate_overall_gini(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_gini =  (p_data_below * calculate_gini(data_below) + p_data_above * calculate_gini(data_above))
    
    return overall_gini
def determine_best_split(data, potential_splits, purity = "gini"):
    if(purity == "entropy"):
        overall_entropy = 9999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = split_data(data, split_column=column_index, split_value=value)
                current_overall_entropy = calculate_overall_entropy(data_below, data_above)

                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
    else:
        overall_gini = 9999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = split_data(data, split_column=column_index, split_value=value)
                current_overall_gini = calculate_overall_gini(data_below, data_above)

                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value
        
    
    return best_split_column, best_split_value
def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, purity = "gini",min_info_gain=-9999, criteria = "mean"):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    potential_splits = get_potential_splits(data)
    split_column, split_value = determine_best_split(data, potential_splits,purity)
    data_below, data_above = split_data(data, split_column, split_value)
    
    if(purity == "entropy"):
        current_info_gain = calculate_entropy(data) - np.mean(calculate_entropy(data_below) + calculate_entropy(data_above))/2
        
    else:
        current_info_gain = calculate_gini(data) - np.mean(calculate_gini(data_below) + calculate_gini(data_above))/2
    
    
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth) or (current_info_gain < min_info_gain):
        classification = classify_data(data, criteria)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
       
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data, criteria)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, purity, min_info_gain,criteria)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, purity, min_info_gain,criteria)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)    