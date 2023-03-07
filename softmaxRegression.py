class SoftMaxRegression:
    """
    SoftMax regression.
    Can be used as the last layer as well. 
    """
    
    W: np.array = None # Weights of this layer 
    b: np.array = None # Bais for the layer
    X: np.array = None # The input X
    Y: np.array = None # The input Y
    A: np.array = None # The product of Z
    isStop:bool = False
    
    
    
    def __init__(self,X ,Y,learning_rate = 0.001,beta1=0.9,beta2=0.999, optimizer = "Adam",decay_rate=0.9):
        self.W = np.random.randn(Y.shape[0],X.shape[0])*(1/np.sqrt(X.shape[0]))
        self.b = np.zeros((Y.shape[0],1))
        self.X = X
        self.Y = Y
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.m = self.Y.shape[0]
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.__initialize_gradients()
            
    def __forward(self):
        Z = np.dot(self.W,self.X) + self.b
        exps = np.exp(Z)
        self.A = exps / np.sum(exps,axis=0,keepdims=True)
        return self.A 
    
    def compute_cost(self):
        cost = (1 / self.m) * np.sum(-1 * np.sum(self.Y * np.log(self.A),keepdims=True))
        if cost < 0.03:
            self.isStop = True
        return cost
        
    def __initialize_gradients(self):
        if self.optimizer == 'Adam':
            self.vdW = np.zeros_like(self.W)
            self.vdb = np.zeros_like(self.b)
            self.sdW = np.zeros_like(self.W)
            self.sdb = np.zeros_like(self.b)
            
        elif self.optimizer == 'momentum':
            self.vdW = np.zeros_like(self.W)
            self.vdb = np.zeros_like(self.b)
            
            
    def __backprop(self):
        
        self.dZ = self.A - self.Y
        self.dW = (1 / self.m) * np.dot(self.dZ,self.X.T)
        self.db = (1 / self.m) * np.sum(self.dZ, axis=1,keepdims=True)
        return np.dot(self.W.T, self.dZ) # dA for the backprob  

    def __update_weights(self):
        
        #Adam optimization
        if self.optimizer == "Adam":
            self.vdW = self.beta1 * self.vdW + (1-self.beta1)*self.dW
            self.vdb = self.beta1 * self.vdb + (1-self.beta1)*self.db
            self.sdW = self.beta2 * self.sdW + (1-self.beta2)*np.square(self.dW)
            self.sdb = self.beta2 * self.sdb + (1-self.beta2)*np.square(self.db)
            self.W = self.W - self.learning_rate * (self.vdW / (np.sqrt(self.sdW)+ 1e-12))
            self.b = self.b - self.learning_rate * (self.vdb / (np.sqrt(self.sdb)+ 1e-12))
        
        #Exponentially weighted average 
        elif self.optimizer == "momentum":
            
            self.vdW = self.beta1 * self.vdW + (1-self.beta1)*self.dW
            self.vdb = self.beta1 * self.vdb + (1-self.beta1)*self.db
            self.W = self.W - self.learning_rate * self.vdW 
            self.b = self.b - self.learning_rate * self.vdb
            
        #batch gradient descent
        else:
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db
            
            
    def train(self,num_of_iterations):
        costs = []
        iterations = [] 
        for i in range(1,num_of_iterations):
            if self.isStop:
                break 
                
            self.__forward()
            self.__backprop()
            self.__update_weights()
            cost = self.compute_cost()

            if i % 500 == 0:
                costs.append(cost)
                print(f"The cost of the {i}th iteration is:= {cost}")
                iterations.append(i)
        
        sns.scatterplot(x=iterations,y=costs)
        plt.show()
        
            
    def predict(self, X_test, Y_test):
        self.X = X_test.copy()
        self.Y = Y_test.copy()
        self.m = self.Y.shape[1]
        self.__forward()
        predictions = np.argmax(self.A,axis=0) == np.argmax(self.Y,axis=0)
        accuracy = predictions.sum() / self.m 
        return accuracy
    
    
    def __str__(self):
        return f"""The shape of weights is {self.W.shape} and is equal = {self.W}\nThe shape of bais is {self.b.shape} and is equal = {self.b}
        \nThe shape of X is {self.X.shape} and is equal = {self.X}\nThe shape of Y is {self.Y.shape} and is equal = {self.Y}"""
