import tensorflow as tf
import threading


class SyncController:
    def __init__(self,optimizer):
        self.model = None
        self.optimizer = optimizer
        self.grads = list()
        self.lock = threading.Lock()


    # a function to initialize our model
    def initialize_model(self,path = None,model =None,constructor  =None)  :
        if path is not None:
            self.model = tf.keras.models.load_model(path)
        elif model is not None:
            self.model = model
        else:
            self.model = constructor.build_model()

    def append_to_grads(self,new_grads,agent_id):
        self.lock.acquire()
        try:
            print('agent ',agent_id,' is adding its grads')
            self.grads.append(new_grads)
            print('finished ',agent_id)
        finally:
            self.lock.release()

    def update_model(self):
        print('updating_model')
        # summing up all the grads
        grads = self.grads[0]
        for i in range(1,len(self.grads)):
            for j in range(grads):
                grads[j] += self.grads[i][j]
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        self.grads = list()
        print('updating finished')





class AsyncController:
    def __init__(self,optimizer):
        self.model = None
        self.optimizer = optimizer
        self.latest_grad = None
        self.lock = threading.Lock()

    # a function to initialize our model
    def initialize_model(self, path=None, model=None, constructor=None):
        if path is not None:
            self.model = tf.keras.models.load_model(path)
        elif model is not None:
            self.model = model
        else:
            self.model = constructor.build_model()

    def update_model(self,new_grads,agent_id,verbose = True):
        returned_model = None
        self.lock.acquire()
        try:
            if verbose:
                print('agent ',agent_id,' is updating the model')
            self.optimizer.apply_gradients(zip(new_grads, self.model.trainable_variables))
            if verbose:
                print('finished ',agent_id)
                returned_model = tf.keras.models.clone_model(self.model)
                returned_model.set_weights(self.model.get_weights())
        finally:
            self.lock.release()
            return returned_model








