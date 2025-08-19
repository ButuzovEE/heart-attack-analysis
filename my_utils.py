from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
def clr(y_test:list, preds:list, name:str):
    """
    Функция принимает предсказания, правильные ответы, название метода
     после чего печатает его в необходимом формате """
    print('*'*40)
    print(name)
    print(metrics.classification_report(y_test, preds))
    print('*'*40)

def grid_plot(x:list, y:list, x_label:str, title:str, y_label='cross_val'):
    """
    Функция принимает на себя названия осей 
    и данные по перебору гиперпараметров,
    Далее печатает график зависимости метрики от параметров"""
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.plot(x, y, 'go-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
def grid_knn( x_tr, y_tr, x_ts, y_ts, k:int ):
    """
    Функция ринимает количество соседей и данные разбыите на трейн - тест
    и печатает график зависимости среднеквадратичной ошибки от количества соседей 
    на тестовых и тренировочных данных"""
    train_MSE = []
    test_MSE = []
    for k in range(1, k):
        KNN = KNeighborsClassifier(n_neighbors=k)
        KNN.fit(x_tr, y_tr)
        y_pred_mse = KNN.predict(x_ts)
        y_train_pred_mse = KNN.predict(x_tr)

        test_MSE.append(metrics.mean_squared_error(y_pred_mse, y_ts))
        train_MSE.append(metrics.mean_squared_error(y_train_pred_mse, y_tr))
        
    plt.figure(figsize=(20, 10))
        
    plt.plot(train_MSE, label="MSE на обучающей выборке")
    plt.plot(test_MSE, label="MSE на тестовой выборке")
    plt.legend(fontsize=20)
    plt.grid()
    plt.xlabel("Число соседей", fontsize=30)
    plt.ylabel("MSE", fontsize=30)
    plt.show()