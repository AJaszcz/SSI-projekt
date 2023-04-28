import numpy as np
import math
from sklearn.metrics import precision_score,recall_score,f1_score

class KNN():
    def __init__(self,n_neighbours=4, representatives=False, collapse=False):
        self.n_neighbours = n_neighbours
        # rzutowanie
        self.representatives = representatives
        self.collapse = collapse
        pass

    def fit(self, X_train, Y_train):
        if self.representatives:
            self.X_train, self.Y_train = create_representatives(X_train,Y_train)
        else:
            self.X_train = X_train
            self.Y_train = Y_train
        if self.collapse:
            self.collapse = True
            self.X_train = self.collapse_datapoints(X_train)
    
    @staticmethod
    def collapse_datapoints(X):
        return np.mean(X,axis=2)

    def predict_datapoint(self, datapoint):
        class_distances = np.empty((0, 2))  # class labels and distances to the datapoint to other datapoints in X_train
        for x, y in zip(self.X_train, self.Y_train):
            dist = self.distance(datapoint, x)
            class_distances = np.append(class_distances, np.array([[y, dist]]), axis=0)
        
        class_distances = class_distances[class_distances[:, 1].argsort()]
        n_classes = class_distances[:self.n_neighbours, 0]

        _, counts = np.unique(n_classes, return_counts=True)  # find the most common class
        ind = np.argmax(counts)
        return int(n_classes[ind])
    
    def predict_datapoint_for_n_in_range(self, datapoint, n_range):
        class_distances = np.empty((0, 2))  # class labels and distances to the datapoint to other datapoints in X_train
        for x, y in zip(self.X_train, self.Y_train):
            dist = self.distance(datapoint, x)
            class_distances = np.append(class_distances, np.array([[y, dist]]), axis=0)
        
        class_distances = class_distances[class_distances[:, 1].argsort()]
        predictions = np.empty((n_range),dtype=int)
        for i in range(n_range):
            n_classes = class_distances[:i+1, 0]
            _, counts = np.unique(n_classes, return_counts=True)  # find the most common class
            ind = np.argmax(counts)
            predictions[i]=int(n_classes[ind])
        return predictions

    @staticmethod
    def distance(datapoint1, datapoint2):  # euclidian distance
        flattened1 = np.ravel(datapoint1)
        flattened2 = np.ravel(datapoint2)
        return np.linalg.norm(flattened1 - flattened2)
    
    def predict(self,X):
        predicted = np.empty(len(X),dtype=int)
        for i in range(len(X)):
            predicted[i]=self.predict_datapoint(X[i])
        return predicted
    
    def predict_for_n_in_range(self,X,n_range):
        predicted = np.empty([len(X),n_range],dtype=int)
        for i in range(len(X)):
            predicted[i]=self.predict_datapoint_for_n_in_range(X[i],n_range)
        return predicted
    
    def score(self,X,y):
        if self.collapse:
            X = self.collapse_datapoints(X)
        correct = 0
        # number of classes
        confusion_matrix = np.zeros([10,10],dtype=int)
        metrics= np.zeros((3),dtype=float)
        #accuracy
        predicted_lables = self.predict(X)
        for predicted,actual in zip(predicted_lables,y):
            if predicted==actual:
                correct+=1
            confusion_matrix[actual][predicted]+=1

        metrics[0] = precision_score(y,predicted_lables,average="macro")
        metrics[1]  = recall_score(y,predicted_lables,average="macro")
        metrics[2] = f1_score(y,predicted_lables,average="macro")
        return correct/len(y), confusion_matrix,metrics
    
    def score_for_n_in_range(self,X,y,n_range):
        if self.collapse:
            X = self.collapse_datapoints(X)
        score = np.zeros(n_range,dtype=np.float32)
        
        # number of classes
        confusion_matrix = np.zeros([n_range,10,10],dtype=int)
        # metrics= np.zeros([n_range,3],dtype=float)
        #accuracy
        i = 0   # im a lazy ass
        for predicted_column in self.predict_for_n_in_range(X,n_range).T:
            correct = 0
            for predicted, actual in zip(predicted_column,y):
                if predicted==actual:
                    correct+=1
                confusion_matrix[i][actual][predicted]+=1   #jak w sklearn
            # metrics[i][0] = precision_score(y,predicted_column,average="macro")
            # metrics[i][1]  = recall_score(y,predicted_column,average="macro")
            # metrics[i][2] = f1_score(y,predicted_column,average="macro")
            score[i]=correct/len(y)
            i+=1
        return score,confusion_matrix
    
    def cross_val_score(self,X,Y, n_folds=4):
        rest = len(X)%n_folds
        if rest!=0:
            print(f"WARNING! Cannot divide given dataset equally into {n_folds} parts. Ignoring last {rest} elements!")
        X_sets = np.split(X[:-rest],n_folds)
        Y_sets = np.split(Y[:-rest],n_folds)

        # nowy model, aby nie ndapisywac fit
        model = KNN(self.n_neighbours, 
                    self.representatives, 
                    self.collapse)
        
        # accuracy
        scores = np.empty(n_folds,dtype=float)
        cm= np.empty([n_folds,10,10],dtype=int)
        metrics = np.zeros([n_folds,3],dtype=float)

        for i in range(n_folds):
            X_train = X_sets.copy()
            Y_train = Y_sets.copy()
            X_val = X_train.pop(i)
            Y_val = Y_train.pop(i)

            model.fit(np.concatenate(X_train),
                      np.concatenate(Y_train))
            scores[i],cm[i],metrics[i] = model.score(X_val,Y_val)
        return scores,cm,metrics

# def create_representatives(X,Y):
#     n_classes = np.zeros(10,dtype=int)  # number of classes
#     for y in Y:
#         n_classes[y]+=1
    
#     X_sorted = [x for _,x in sorted(zip(Y,X),key=lambda el : el[0])]    # sortuje wg etykiet
    
#     index_x=0
#     class_representants = []
#     for n in n_classes:
#         n_representatives = math.ceil(math.log2(n))
#         representatives = np.empty((n_representatives),dtype=type(X_sorted[0]))
#         #todo fix
#         for i in range(n_representatives):
#             print(index_x+(i+1)*n_representatives-(index_x+i*n_representatives))
#             representatives[i]=sum(X_sorted[index_x+i*n_representatives     :    index_x+(i+1)*n_representatives])//n_representatives
#         #representatives[n_representatives-1] = sum(X_sorted[index_x+(n_representatives-1)*n_representatives     :   index_x+n_representatives*n_representatives+(n_representatives*n_representatives-n)])//(n_representatives*n_representatives-n)
#         index_x+=n
#         for el in representatives:
#             class_representants.append(el)
#     return np.array(class_representants,dtype=int), np.ravel([np.full((math.ceil(math.log2(n))),i,dtype=int) for i in range(10)])  # wtf is this piece of shit

def create_representatives(X,Y):
    n_classes = np.zeros(10,dtype=int)  # number of classes
    for y in Y:
        n_classes[y]+=1
    
    X_sorted = [x for _,x in sorted(zip(Y,X),key=lambda el : el[0])]    # sortuje wg etykiet
    
    index_x=0
    class_representants = []
    for n in n_classes:
        n_representatives = math.ceil(math.log2(n))
        representatives = np.empty((n_representatives),dtype=type(X_sorted[0]))
        #todo fix
        step = math.ceil(n/n_representatives)
        for i in range(n_representatives-1):
            #print(index_x+(i+1)*step-(index_x+i*step))
            representatives[i]=sum(X_sorted[index_x+i*step     :    index_x+(i+1)*step])//step
        representatives[n_representatives-1] = sum(X_sorted[index_x+(n_representatives-1)*step     :    index_x+n])//(n-(n_representatives-1)*step)
        #representatives[n_representatives-1] = sum(X_sorted[index_x+(n_representatives-1)*n_representatives     :   index_x+n_representatives*n_representatives+(n_representatives*n_representatives-n)])//(n_representatives*n_representatives-n)
        index_x+=n
        for el in representatives:
            class_representants.append(el)
    return np.array(class_representants,dtype=int), np.ravel([np.full((math.ceil(math.log2(n))),i,dtype=int) for i in range(10)])  # wtf is this piece of shit