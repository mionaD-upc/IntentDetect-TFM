import numpy as np
from hyperopt import base
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing, any_regressor
from sklearn.metrics import accuracy_score,confusion_matrix
from hyperopt import tpe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from graphviz import Digraph
from datetime import datetime
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import json
from sklearn.metrics import mean_absolute_error
from sklearn import datasets
import shutil


visualisation =''

def preprocess(data_file_path):
    file_name = os.path.basename(data_file_path)

    df = pd.read_csv(data_file_path)
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
            df[column] = LabelEncoder().fit_transform(df[column])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    StandardScalerModel = StandardScaler()
    X = StandardScalerModel.fit_transform(X)

    data = np.column_stack((X, y))
    column_names = df.columns
    res = pd.DataFrame(data,columns =column_names)
    for column in res.columns:
        res[column] = pd.to_numeric(res[column])
    
    preprocessed_file_path = f'uploads/preprocessed/{file_name}'
    if not os.path.exists('uploads/preprocessed'):
            os.makedirs('uploads/preprocessed')

    res.to_csv(preprocessed_file_path, index=False)

    return X, y


def pipeline_generator(data_file_path, intent):

    X, y = preprocess(data_file_path=data_file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=34)
    dataset_name = os.path.basename(data_file_path).split('.')[0]

    
    if intent =='classification':
         
        # Instantiate a HyperoptEstimator with the search space and number of evaluations
        estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                                preprocessing=any_preprocessing("my_pre"),
                                loss_fn=accuracy_score,
                                algo=tpe.suggest,
                                max_evals=5,
                                trial_timeout=300,  verbose= False)
        estim.fit(X_train, y_train)
        metric_file_name = f"accuracy_{dataset_name}.json"
        metric_name = "accuracy"



    elif intent =='regression':
        # Instantiate a HyperoptEstimator with the search space and number of evaluations
        estim = HyperoptEstimator(regressor=any_regressor("my_reg"),
                                preprocessing=any_preprocessing("my_pre"),
                                loss_fn=mean_absolute_error,
                                algo=tpe.suggest,
                                max_evals=5,
                                trial_timeout=300,  verbose= False)

        estim.fit(X_train, y_train)


        metric_file_name = f"mean_absolute_error_{dataset_name}.json"
        metric_name = "mae"
    
    
    metric_value = estim.score(X_test, y_test)

    # data = {
    #         "dataset_name": dataset_name,
    #         "intent":intent,
    #         "metric_name": metric_name,
    #         "metric_value": metric_value
    # }
    # if not os.path.exists('static/hyperopt-results/metric'):
    #         os.makedirs('static/hyperopt-results/metric')

    # # Write data to JSON file
    # with open(f'static/hyperopt-results/metric/{metric_file_name}', "w") as json_file:
    #     json.dump(data, json_file)

    # print(estim.best_model())

    y_pred = estim.predict(X_test)
    pipeline= estim.best_model()

    if intent =='classification':
        visualisation = 'Confusion Matrix';
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if os.path.exists('static/hyperopt-results/images'):
            shutil.rmtree('static/hyperopt-results/images')
        os.makedirs('static/hyperopt-results/images')



        img_filename = f"static/hyperopt-results/images/{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}-conf_matrix.png"

        plt.savefig(img_filename)

    elif intent =='regression':
        visualisation = 'Scatter Plot';
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Values')
        if os.path.exists('static/hyperopt-results/images'):
            shutil.rmtree('static/tpot-results/images')
        os.makedirs('static/hyperopt-results/images')
        
        img_filename = f"static/hyperopt-results/images/{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}-scatter_plot.png"
        plt.savefig(img_filename)

    # Generate workflow visualization and save it to a file:

    if not os.path.exists('static/hyperopt-results/dataflows'):
        os.makedirs('static/hyperopt-results/dataflows')

    graph_filename = f"static/hyperopt-results/dataflows/{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}-dataflow"
    graph = Digraph('DataFlow', filename=graph_filename)
    graph.attr(rankdir='LR')

    # Create rectangles
    graph.node('Dataset', fillcolor='orange', label=f'Dataset:\n{dataset_name}.csv')
    graph.node('Visualization', fillcolor='lightgreen', label=f'Visualization:\n{visualisation}')
    algo = pipeline['learner']
    algo_name = str(algo).split('(')[0]
    graph.node('Algorithm', fillcolor='lightblue', label=f'Algorithm:\n{algo_name}')

    if len(pipeline['preprocs']) != 0:
        prepro = pipeline['preprocs'][0]
        prepro_name = str(prepro).split('(')[0]
        graph.node('Preprocessing', fillcolor='lightblue', label=f'Preprocessing:\n{prepro_name}')

    # Add edges
    if len(pipeline['preprocs']) != 0:
        path='example/template-4-dataflow.svg'
        
        if os.path.exists(path):
            graph_path = graph_filename + '.svg'
            # print(f"The path '{path}' exists.")
            change = open(path, "rt")
            data = change.read()
            data = data.replace('dataset_name.csv', dataset_name + '.csv')
            data = data.replace('methodX', prepro_name)
            data = data.replace('Scatter Plot/Confusion Matrix', visualisation)
            data = data.replace('Classifier/Regressor', algo_name)
            change.close()
            change = open(graph_path, "wt")
            change.write(data)
            change.close()
        else:
            print(f"The path '{path}' does not exist.")
        graph.edge('Dataset', 'Preprocessing')
        graph.edge('Preprocessing', 'Algorithm')
        graph.edge('Algorithm', 'Visualization')
    else:
        path='example/template-3-dataflow.svg'
        
        if os.path.exists(path):
            print(f"The path '{path}' exists.")
            graph_path = graph_filename + '.svg'
            # print(f"The path '{path}' exists.")
            change = open(path, "rt")
            data = change.read()
            data = data.replace('dataset_name.csv', dataset_name + '.csv')
            data = data.replace('Scatter Plot/Confusion Matrix', visualisation)
            data = data.replace('Classifier/Regressor', algo_name)
            change.close()
            change = open(graph_path, "wt")
            change.write(data)
            change.close()
        else:
            print(f"The path '{path}' does not exist.")
        graph.edge('Dataset', 'Algorithm')
        graph.edge('Algorithm', 'Visualization')

    graph.save()

    return img_filename, graph_filename + '.svg', metric_name, metric_value



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_path of csv file> --intent <regression/classification>")
        sys.exit(1)

    file_path = sys.argv[1]
    intent= sys.argv[3]
    pipeline_generator(file_path, intent)