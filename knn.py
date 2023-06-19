import numpy as np
import math
from sklearn.metrics import precision_score,recall_score,f1_score

class KNN():
    def __init__(self,n_neighbours=5, representatives=False, project=False):
        self.n_neighbours = n_neighbours
        self.representatives = representatives
        self.project = project
        pass

    def fit(self, X_train, Y_train):
        if self.representatives:
            self.X_train, self.Y_train = create_representatives(X_train,Y_train)
        else:
            self.X_train = X_train
            self.Y_train = Y_train
        if self.project:
            self.project = True
            self.X_train = self.project_datapoints(X_train)
    
    @staticmethod
    def project_datapoints(X):
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
    
    def predict_for_n_in_range(self,X,n_range): #
        predicted = np.empty([len(X),n_range],dtype=int)
        for i in range(len(X)):
            predicted[i]=self.predict_datapoint_for_n_in_range(X[i],n_range)
        return predicted
    
    def score(self,X,y):
        if self.project:
            X = self.project_datapoints(X)
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
        if self.project:
            X = self.project_datapoints(X)
        score = np.zeros(n_range,dtype=np.float32)
        
        # creates empty cm with fixed 10 number of classes
        confusion_matrix = np.zeros([n_range,10,10],dtype=int)

        #accuracy score
        i = 0 
        for predicted_column in self.predict_for_n_in_range(X,n_range).T:
            correct = 0
            for predicted, actual in zip(predicted_column,y):
                if predicted==actual:
                    correct+=1
                confusion_matrix[i][actual][predicted]+=1   # the same way sklearn produces CM
            score[i]=correct/len(y)
            i+=1
        return score,confusion_matrix
    
    def cross_val_score(self,X,Y, n_folds=4):
        rest = len(X)%n_folds
        if rest!=0:
            print(f"WARNING! Cannot divide given dataset equally into {n_folds} parts. Ignoring last {rest} elements!")
        X_sets = np.split(X[:-rest],n_folds)
        Y_sets = np.split(Y[:-rest],n_folds)

        # create new model, not to overwrite current fit
        model = KNN(self.n_neighbours, 
                    self.representatives, 
                    self.project)
        
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

def create_representatives(X,Y):
    n_classes = np.zeros(10,dtype=int)  # number of classes
    for y in Y:
        n_classes[y]+=1

    # sorts accoring to labels
    X_sorted = [x for _,x in sorted(zip(Y,X),key=lambda el : el[0])]
    
    index_x=0
    class_representants = []
    for n in n_classes:
        n_representatives = math.ceil(math.log2(n)) # eq. 1
        representatives = np.empty((n_representatives),dtype=type(X_sorted[0]))
        step = math.ceil(n/n_representatives) # eq. 2
        for i in range(n_representatives-1):
            representatives[i]=sum(X_sorted[index_x+i*step     :    index_x+(i+1)*step])//step
        representatives[n_representatives-1] = sum(X_sorted[index_x+(n_representatives-1)*step     :    index_x+n])//(n-(n_representatives-1)*step) # eq. 4
        index_x+=n
        for el in representatives:
            class_representants.append(el)
    return np.array(class_representants,dtype=int), np.ravel([np.full((math.ceil(math.log2(n))),i,dtype=int) for i in range(10)])